import argparse
import json
import logging
import re
import requests
from typing import List, Dict
import os
import asyncio
import tqdm

# Import sglang libraries
import sglang as sgl
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import Tool, Function

# Import locally defined search API modules
from search_api_local import SearchAPILocal
# from search_api_searxng import SearchAPISearxng
from search_api_serper import SearchAPISerper as SearchAPISearxng


# --- Command-line Argument Parsing ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch inference script for fact-checking model using SGLang."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the SGLang-compatible model.",
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to save the output JSONL file.",
    )
    parser.add_argument(
        "--gpu_idx", type=int, default=None, help="GPU index to use."
    )
    parser.add_argument(
        "--mode", type=str, required=True, 
        # [MODIFIED] Only 'retrieval' mode is supported
        choices=["retrieval"], 
        help="Set operating mode: 'retrieval' (claim-level search-based)."
    )
    parser.add_argument(
        "--scheme", type=str, default="scoring", 
        # [MODIFIED] Only 'scoring' scheme is supported
        choices=["scoring"],
        help="Evaluation scheme: 'scoring' (judge correctness of each claim)."
    )
    parser.add_argument(
        "--disable_thinking", action="store_true", default=False,
        help="Disable the model's thinking process."
    )
    parser.add_argument(
        "--max_token", type=int, default=2048, help="Maximum new tokens to generate."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature for generation."
    )
    parser.add_argument(
        "--disable_cache_for_serper", action="store_true", default=False,
    )
    return parser.parse_args()


# --- Tool Definition ---
SEARCH_TOOL_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "search_local",
            "description": "Searches the local Wikipedia document database. Use this for well-established factual knowledge, definitions, and historical information.",
            "parameters": {
                "type": "object",
                "properties": { "query": { "type": "string", "description": "The search query keyword for the local Wikipedia database." } },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Performs a web search using Google. Use this for recent events, current affairs, opinions, or information not found in the local database.",
            "parameters": {
                "type": "object",
                "properties": { "query": { "type": "string", "description": "The search query keyword for the web search engine." } },
                "required": ["query"],
            },
        },
    }
]

# --- Helper Functions (Extraction & Evaluation) ---
def extract_final_scoring(model_generated_output: str) -> int:
    """Extracts the final score from the model's output."""
    if not isinstance(model_generated_output, str): return 0
    match = re.search(r"<verdict>\s*(\d+)\s*</verdict>", model_generated_output, re.IGNORECASE)
    if not match:
        match = re.search(r"\*\*Final Verdict\*\*:\s*(\d+)\s*", model_generated_output, re.IGNORECASE)
    if not match:
        match = re.search(r"Final Verdict:\s*(\d+)\s*", model_generated_output, re.IGNORECASE)
    try:
        return int(match.group(1))
    except:         
        return 0

def check_ready_for_evaluation(model_generated_output: str) -> bool:
    """Checks if the model is ready to move to the final evaluation stage."""
    # [MODIFIED] Simplified check
    return "READY_FOR_EVALUATION" in model_generated_output or "READY_FOR_ANSWERING" in model_generated_output

def batched_sglang_generation(input_ids, sampling_params, engine, BATCH_SIZE=100):
    """Generates text in batches using the SGLang engine."""
    batched_input_ids = [input_ids[i:i + BATCH_SIZE] for i in range(0, len(input_ids), BATCH_SIZE)]
    results = []
    for batch in tqdm.tqdm(batched_input_ids, desc="Batched Generating"):
        results.extend(engine.generate(input_ids=batch, sampling_params=sampling_params))
    return results


# --- [NEW] Claim Decomposition Function ---
def perform_claim_decomposition(input_data: List[Dict], engine, tokenizer, args) -> List[Dict]:
    """
    [REWRITTEN]
    Decomposes each answer in each item into claims, and returns a new 
    list of items where the 'answers' list is replaced by the claims.
    """
    print("--- Starting Claim Decomposition Stage ---")
    
    decomposition_prompt_template = """You are an expert in claim decomposition. Your task is to break down the given "Answer" into a list of simple, verifiable, and self-contained claims based on the "Question".
Question: {question}
Answer: {answer}
Please provide the decomposed claims as a JSON list of strings.
Example:
["Claim 1", "Claim 2", "Claim 3"]
Claims:
"""
    input_ids = []
    task_metadata = [] # To map responses back to (item_index, answer_index)
    
    # 1. Prepare all decomposition tasks
    for i, item in enumerate(tqdm.tqdm(input_data, desc="Preparing decomposition")):
        if "answers" not in item: continue
        for j, ans_dict in enumerate(item["answers"]):
            ans_text = ans_dict.get("answer", "")
            if not ans_text: continue
            
            prompt = decomposition_prompt_template.format(question=item['question'], answer=ans_text)
            messages = [{"role": "user", "content": prompt}]
            input_ids.append(tokenizer.apply_chat_template(
                conversation=messages, tokenize=True, add_generation_prompt=True
            ))
            task_metadata.append({
                "original_item_index": i, 
                "original_answer_id": j, # This is the index (0, 1, 2...)
                "original_item_question": item['question']
            })

    # 2. Run batched decomposition
    sampling_params = {"max_new_tokens": 1024, "temperature": 0.0}
    print(f"Decomposing {len(input_ids)} answers into claims...")
    responses = batched_sglang_generation(input_ids, sampling_params, engine)
    
    # 3. Create a new data structure to hold the decomposed items
    # We aggregate claims back into their original items
    temp_item_storage = {} # Key: original_item_index
    
    for meta, response in zip(task_metadata, responses):
        original_item_index = meta["original_item_index"]
        original_answer_id = meta["original_answer_id"] # e.g., 0 or 1
        
        # Initialize the new item if it's the first time we see it
        if original_item_index not in temp_item_storage:
            temp_item_storage[original_item_index] = {
                "question": meta["original_item_question"],
                "answers": [], # This will hold the new claim dicts
                # Copy other fields from the original item (like 'id')
                **{k: v for k, v in input_data[original_item_index].items() if k not in ['question', 'answers']}
            }

        # Parse claims from the response
        generated_text = response["text"]
        claims = []
        try:
            # Try to parse as JSON list
            json_match = re.search(r'\[.*\]', generated_text, re.DOTALL)
            if json_match:
                claims = json.loads(json_match.group(0))
            if not isinstance(claims, list) or not claims:
                raise json.JSONDecodeError("Not a valid list")
        except (json.JSONDecodeError, TypeError):
            # Fallback: split by newline, filter out junk
            claims = [line.strip().lstrip('-* ').replace('"', '') for line in generated_text.split('\n') if line.strip() and len(line.strip()) > 5]
            if not claims: # Final fallback
                claims = [generated_text.strip()]
        
        # Add these claims to the new "answers" list
        for claim_text in claims:
            if not claim_text: continue
            
            new_answer_dict = {
                "answer": claim_text, # The claim is the new answer
                "original_answer_id": original_answer_id # As requested
            }
            temp_item_storage[original_item_index]["answers"].append(new_answer_dict)
    
    # 4. Convert the temp storage dict to the final list, preserving order
    final_new_input_data = [temp_item_storage[i] for i in sorted(temp_item_storage.keys())]
            
    print(f"--- Claim Decomposition Finished: {len(input_data)} items processed ---")
    return final_new_input_data


# --- Agent State Processing Functions ---

def process_decision_stage(jobs_to_decide, engine, tokenizer, parser, args):
    """
    Decides the next action (search or evaluate) based on the conversation history.
    """
    if not jobs_to_decide: return
    
    print(f"--- Deciding next action for {len(jobs_to_decide)} jobs ---")
    
    input_ids = [
        tokenizer.apply_chat_template(
            conversation=job["messages"], 
            tokenize=True, 
            add_generation_prompt=True, 
            tools=SEARCH_TOOL_DEFINITION,
            enable_thinking=(not args.disable_thinking)
        ) for job in jobs_to_decide
    ]

    sampling_params = {"max_new_tokens": args.max_token, "temperature": args.temperature}
    responses = batched_sglang_generation(input_ids, sampling_params, engine)
    
    for job, response in zip(jobs_to_decide, responses):
        generated_text = response["text"]
        job["messages"].append({"role": "assistant", "content": generated_text})

        if check_ready_for_evaluation(generated_text):
            job["current_step"] = "evaluate"
            continue
        
        try:
            _, calls = parser.parse_non_stream(generated_text)
            job["current_step"] = "tool_execution" if calls else "evaluate"
            job["tool_calls"] = calls
        except (json.JSONDecodeError, ValueError):
            job["current_step"] = "evaluate" # Fallback to evaluation on parsing failure

def process_tool_execution_stage(jobs_with_calls, local_api, searxng_api):
    """Executes the tool calls, adds result to history, and sets state to decision."""
    if not jobs_with_calls: return
    print(f"--- Executing tools for {len(jobs_with_calls)} jobs ---")

    local_tasks, web_tasks = [], []
    for job in jobs_with_calls:
        if not job.get("tool_calls"): continue
        call = job["tool_calls"][0] # Process one call per turn
        try:
            params = json.loads(call.parameters) if isinstance(call.parameters, str) else call.parameters
            query = params.get("query")
            if not query: continue
            if call.name == "search_local": local_tasks.append({"job": job, "query": query})
            elif call.name == "search_web": web_tasks.append({"job": job, "query": query})
        except (json.JSONDecodeError, AttributeError):
            logging.warning(f"Job {job['id']}: Failed to parse parameters for tool call ({call.name})")
            job["current_step"] = "decision" # Go back to decision if tool call fails
            continue

    if local_tasks:
        results = local_api.search_api_call([t["query"] for t in local_tasks])
        for task, result in zip(local_tasks, results):
            content = f"[Source: Local Wikipedia]\n{result}"
            task["job"]["messages"].append({"role": "tool", "content": content})
            task["job"]["current_step"] = "decision"
            task["job"]["search_count"] += 1

    if web_tasks:
        results = searxng_api.search_api_call([t["query"] for t in web_tasks])
        for task, result in zip(web_tasks, results):
            content = f"[Source: Google Search]\n{result}"
            task["job"]["messages"].append({"role": "tool", "content": content})
            task["job"]["current_step"] = "decision"
            task["job"]["search_count"] += 1
            
    # Clean up tool_calls after execution
    for job in jobs_with_calls:
        if "tool_calls" in job:
            del job["tool_calls"]

# --- Main Function ---
def main():
    """Main execution function."""
    args = parse_args()
    MAX_TURNS = 5

    # --- [MODIFIED] Assertions for simplified script ---
    assert args.scheme == "scoring", "This script is modified to only support 'scoring' scheme."
    assert args.mode == "retrieval", "This script is modified to only support 'retrieval' mode."
    
    print(f"Arguments: {args}")
    print("Initializing SGLang engine...")
    engine = sgl.Engine(model_path=args.model_path, trust_remote_code=True, mem_fraction_static=0.8)
    tokenizer = engine.tokenizer_manager.tokenizer
    tools = [Tool(type=t["type"], function=Function(**t["function"])) for t in SEARCH_TOOL_DEFINITION]
    parser = FunctionCallParser(tools=tools, tool_call_parser="qwen25")

    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f if line.strip()]

    # --- [NEW] Claim Decomposition Step ---
    # Replaces 'answers' list in input_data with decomposed 'claims'
    input_data = perform_claim_decomposition(input_data, engine, tokenizer, args)
    # 'input_data' now contains items with a list of claims as "answers"

    # --- Prompt Templates ---
    # [MODIFIED] Prompt now includes a placeholder for the answer (claim)
    scoring_search_prompt = f"""Your task is to gather information to verify the factuality of a claim to a question. Based on the question and claim, think and identify what information you need and generate search queries using the available tools.

Question: {{question}}
Claim: {{answer}}

Leverage both tools (`search_local` for Wikipedia, `search_web` for Google) and generate one search query per turn. If you have enough information, respond with "READY_FOR_EVALUATION"."""

    # --- [MODIFIED] Job Initialization ---
    # We now create one job PER CLAIM, not one job per question.
    jobs = []
    print(f"Initializing jobs for {len(input_data)} items...")
    
    claim_count = 0
    for i, item in enumerate(input_data): # i is original_item_index
        if "answers" not in item:
            continue
            
        for j, answer_dict in enumerate(item["answers"]): # j is original_answer_index
            claim_text = answer_dict["answer"]
            
            # This is the prompt you requested, with both question and answer
            content = scoring_search_prompt.format(
                question=item["question"],
                answer=claim_text
            )
            
            job = {
                "id": f"{i}_{j}", # Unique ID for the claim
                "original_item_index": i,
                "original_answer_index": j,
                "original_item_question": item["question"],
                "original_answer_text": claim_text,
                "original_answer_id": answer_dict.get("original_answer_id"), # From decomposition
                "search_count": 0,
                "messages": [{"role": "user", "content": content}],
                "current_step": "decision"
            }
            jobs.append(job)
            claim_count += 1

    print(f"Created {len(jobs)} jobs for {claim_count} total claims.")

    # --- Agentic Loop (Decide -> Execute) ---
    print("Initializing search APIs: Local (Wiki) and Searxng (Google)...")
    local_api = SearchAPILocal()
    searxng_api = SearchAPISearxng(use_cache=(not args.disable_cache_for_serper))

    for turn in tqdm.tqdm(range(MAX_TURNS), desc="Agent Turns"):
        # Stage 1: DECIDE - Decide next action for jobs in 'decision' state
        jobs_to_decide = [j for j in jobs if j.get("current_step") == "decision"]
        if not jobs_to_decide:
            print(f"No more jobs in 'decision' state at turn {turn+1}.")
        else:
            process_decision_stage(jobs_to_decide, engine, tokenizer, parser, args)

        # Stage 2: EXECUTE - Run tool calls generated in the decision stage
        jobs_to_execute = [j for j in jobs if j.get("current_step") == "tool_execution"]
        if jobs_to_execute:
            process_tool_execution_stage(jobs_to_execute, local_api, searxng_api)
        
        active_jobs = [j for j in jobs if j.get("current_step") not in ["evaluate", "done"]]
        if not active_jobs:
            print(f"All jobs completed or moved to evaluation by turn {turn + 1}. Exiting agent loop.")
            break
    
    # Force remaining jobs to evaluation
    for job in jobs:
        if job.get("current_step") not in ["evaluate", "done"]:
            job["current_step"] = "evaluate"
    print("--- Agentic loop finished ---")

    print("All sequences processed. Re-aggregating and saving results...")
    
    # [MODIFIED] Re-aggregate jobs back into items
    temp_item_storage = {}
    
    # Sort jobs to ensure answers are re-assembled in the correct order
    for job in sorted(jobs, key=lambda x: (x["original_item_index"], x["original_answer_index"])):
        item_idx = job["original_item_index"]
        
        # If this is the first time seeing this item, initialize it
        if item_idx not in temp_item_storage:
            # Copy non-answer fields from the original source item
            original_source_item = input_data[item_idx]
            temp_item_storage[item_idx] = {
                k: v for k, v in original_source_item.items() if k != 'answers'
            }
            temp_item_storage[item_idx]["answers"] = [] # Prepare to hold new answer dicts

        # Create the new answer dict containing all results for this claim
        new_answer_dict = {
            "answer": job["original_answer_text"],
            "original_answer_id": job.get("original_answer_id"),
            "verdict_response": job.get("verdict_response", "[EVALUATION FAILED]"),
            "predicted_scoring": job.get("predicted_scoring", 0),
            "retrieval_path": job.get("messages", []), # This is the per-claim retrieval
            "search_count": job.get("search_count", 0)  # This is the per-claim search count
        }
        
        temp_item_storage[item_idx]["answers"].append(new_answer_dict)

    # Convert the storage dict back to the final list
    final_results = [temp_item_storage[i] for i in sorted(temp_item_storage.keys())]

    # Write the results to the output file
    output_path = args.output_file
    with open(output_path, "w", encoding="utf-8") as f_out:
        for result in final_results:
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Results saved to {output_path}")

    # [MODIFIED] Updated print statement
    print("Scoring mode finished. Each claim has been retrieved for and scored individually.")

if __name__ == "__main__":
    main()