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

# Import metrics libraries
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau

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
        "--mode", type=str, required=True, choices=["retrieval", "direct_gen"],
        help="Set operating mode: 'retrieval' (search-based) or 'direct_gen' (direct generation)."
    )
    parser.add_argument(
        "--scheme", type=str, default="ranking", choices=["ranking", "pointwise"],
        help="Evaluation scheme: 'ranking' (rank a list of answers) or 'pointwise' (judge correctness of each answer)."
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
    return parser.parse_args()


# --- Tool Definition ---
# Define two separate search tools for the model to choose from
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
def extract_final_ranking(model_generated_output: str) -> List[int]:
    """Extracts the final ranked list from the model's output."""
    if not isinstance(model_generated_output, str): return []
    matches = re.findall(r"<verdict>(.*?)</verdict>", model_generated_output, re.IGNORECASE | re.DOTALL)
    if not matches: return []
    try:
        parts = [part.strip() for part in matches[-1].strip().split('>')]
        return [int(re.search(r'Answer(\d+)', part, re.IGNORECASE).group(1)) - 1 for part in parts if re.search(r'Answer(\d+)', part, re.IGNORECASE)]
    except (ValueError, IndexError, AttributeError):
        return []

def check_ready_for_evaluation(model_generated_output: str) -> bool:
    """Checks if the model is ready to move to the final evaluation stage."""
    return "READY_FOR_EVALUATION" in model_generated_output or "READY_FOR_ANSWERING" in model_generated_output

def evaluate_final_results_ranking(results: List[Dict]):
    """
    Calculates and prints evaluation metrics for the ranking scheme.
    The ground truth is a ranked list. Metrics used are P@1, Kendall's Tau, and NDCG.
    """
    kendall_tau_scores, ndcg_scores = [], []
    top_1_correct_count, valid_evaluation_count, invalid_predictions = 0, 0, 0
    total_items = len(results)
    
    for item in results:
        true_ranking = item.get("verify_result")
        pred_ranking = item.get("predicted_ranking")
        
        is_true_label_valid = isinstance(true_ranking, list) and true_ranking
        if not is_true_label_valid: continue

        is_pred_valid = isinstance(pred_ranking, list) and pred_ranking
        if not is_pred_valid or len(true_ranking) != len(pred_ranking):
            invalid_predictions += 1
            continue

        valid_evaluation_count += 1
        num_answers = len(true_ranking)

        if true_ranking[0] == pred_ranking[0]:
            top_1_correct_count += 1

        true_ranks = [0] * num_answers
        for rank, item_idx in enumerate(true_ranking):
            true_ranks[item_idx-1] = rank  # 存储每个 item_idx 对应的排名

        pred_ranks = [0] * num_answers
        for rank, item_idx in enumerate(pred_ranking):
            pred_ranks[item_idx-1] = rank  # 存储每个 item_idx 对应的排名

        tau, _ = kendalltau(true_ranks, pred_ranks)
        kendall_tau_scores.append(tau)

    if valid_evaluation_count == 0:
        logging.error("Evaluation failed. No valid items to evaluate.")
        return None

    precision_at_1 = top_1_correct_count / valid_evaluation_count if valid_evaluation_count else 0.0
    avg_kendall_tau = sum(kendall_tau_scores) / len(kendall_tau_scores) if kendall_tau_scores else 0.0
    # avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    invalid_ratio = invalid_predictions / total_items if total_items > 0 else 0.0

    metrics_dict = {
        "precision_at_1": round(precision_at_1, 4),
        "average_kendall_tau": round(avg_kendall_tau, 4),
        # "average_ndcg": round(avg_ndcg, 4),
        "invalid_prediction_ratio": round(invalid_ratio, 4),
    }

    print("\n--- Ranking Evaluation Results ---")
    for key, value in metrics_dict.items(): print(f"{key.replace('_', ' ').title()}: {value}")
    print("----------------------------------\n")
    return metrics_dict

def batched_sglang_generation(input_ids, sampling_params, engine, BATCH_SIZE=100):
    """Generates text in batches using the SGLang engine."""
    batched_input_ids = [input_ids[i:i + BATCH_SIZE] for i in range(0, len(input_ids), BATCH_SIZE)]
    results = []
    for batch in tqdm.tqdm(batched_input_ids, desc="Batched Generating"):
        results.extend(engine.generate(input_ids=batch, sampling_params=sampling_params))
    return results

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


def process_evaluation_stage(evaluate_jobs, engine, tokenizer, args):
    """Processes jobs ready for final evaluation and verdict."""
    if not evaluate_jobs: return
    
    print(f"--- Starting final evaluation for {len(evaluate_jobs)} jobs ---")

    sampling_params = {"max_new_tokens": args.max_token, "temperature": args.temperature}

    if args.scheme == "ranking":
        ranking_verdict_prompt_retrieval = """Based on the preceding conversation, your task is to rank all the given answers from most to least factually correct.

Question: {question}
{answers_block}

Please first provide your explanation, and then state your final verdict in the format: '**Final Verdict**: <verdict> AnswerX > AnswerY > AnswerZ </verdict>'. Example: '**Final Verdict**: <verdict> Answer3 > Answer1 > Answer2 </verdict>'."""

        ranking_verdict_prompt_direct = """You are an expert fact-checking assistant. Your task is to rank all the given answers from most to least factually correct based on your internal knowledge.

Question: {question}
{answers_block}

Please first provide your explanation, and then state your final verdict in the format: '**Final Verdict**: <verdict> AnswerX > AnswerY > AnswerZ </verdict>'. Example: '**Final Verdict**: <verdict> Answer3 > Answer1 > Answer2 </verdict>'."""

        verdict_input_ids = []
        for job in evaluate_jobs:
            answers = job["original_item"]["answers"]
            answers_block = "\n".join([f"Answer{i+1}: {ans['answer']}" for i, ans in enumerate(answers)])
            
            if args.mode == "retrieval":
                prompt = ranking_verdict_prompt_retrieval.format(question=job["original_item"]["question"], answers_block=answers_block)
                final_messages = job["messages"] + [{"role": "user", "content": prompt}]
            else: # direct_gen mode
                prompt = ranking_verdict_prompt_direct.format(question=job["original_item"]["question"], answers_block=answers_block)
                final_messages = [{"role": "user", "content": prompt}]
            
            verdict_input_ids.append(tokenizer.apply_chat_template(
                conversation=final_messages, tokenize=True, add_generation_prompt=True, enable_thinking=(not args.disable_thinking)))
        
        verdict_responses = batched_sglang_generation(input_ids=verdict_input_ids, sampling_params=sampling_params, engine=engine)
        
        for job, response in zip(evaluate_jobs, verdict_responses):
            generated_text = response["text"]
            job["verdict_response"] = generated_text
            job["predicted_ranking"] = extract_final_ranking(generated_text)

# --- Main Function ---
def main():
    """Main execution function."""
    args = parse_args()
    MAX_TURNS = 5

    # --- Initialization ---
    if args.scheme == "pointwise" and args.mode == "direct_gen":
        print("Error: 'pointwise' scheme does not support 'direct_gen' mode.")
        exit(1)

    print(f"Arguments: {args}")
    print("Initializing SGLang engine...")
    engine = sgl.Engine(model_path=args.model_path, trust_remote_code=True, mem_fraction_static=0.8)
    tokenizer = engine.tokenizer_manager.tokenizer
    tools = [Tool(type=t["type"], function=Function(**t["function"])) for t in SEARCH_TOOL_DEFINITION]
    parser = FunctionCallParser(tools=tools, tool_call_parser="qwen25")

    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f if line.strip()]

    # --- Prompt Templates ---
    ranking_search_prompt = f"""Your task is to determine the correct factual ranking of the provided answers. Based on the question and answers, think and identify what information you need and generate search queries using the available tools.

Question: {{question}}
{{answers_block}}

Leverage both tools (`search_local` for Wikipedia, `search_web` for Google) and generate one search query per turn. If you have enough information, respond with "READY_FOR_EVALUATION"."""

    pointwise_search_prompt = f"""Your task is to gather information to verify an answer to a question. Based on the question, think and identify what information you need and generate search queries using the available tools.

Question: {{question}}

Leverage both tools (`search_local` for Wikipedia, `search_web` for Google) and generate one search query per turn. If you have enough information, respond with "READY_FOR_ANSWERING"."""

    # --- Job Initialization ---
    jobs = []
    for i, item in enumerate(input_data):
        job_base = {"id": i, "original_item": item, "search_count": 0, "search_results": []}
        if args.mode == "direct_gen":
            jobs.append({**job_base, "current_step": "evaluate"})
            continue

        if args.scheme == "pointwise":
            content = pointwise_search_prompt.format(question=item["question"])
        else: # ranking
            answers_block = "\n".join([f"Answer{i+1}: {ans['answer']}" for i, ans in enumerate(item.get("answers", []))])
            content = ranking_search_prompt.format(question=item["question"], answers_block=answers_block)
        
        jobs.append({**job_base, "messages": [{"role": "user", "content": content}], "current_step": "decision"})

    # --- Agentic Loop (Decide -> Execute) ---
    if args.mode == "retrieval":
        print("Initializing search APIs: Local (Wiki) and Searxng (Google)...")
        local_api = SearchAPILocal()
        searxng_api = SearchAPISearxng()

        for turn in tqdm.tqdm(range(MAX_TURNS), desc="Agent Turns"):
            # Stage 1: DECIDE - Decide next action for jobs in 'decision' state
            jobs_to_decide = [j for j in jobs if j.get("current_step") == "decision"]
            process_decision_stage(jobs_to_decide, engine, tokenizer, parser, args)

            # Stage 2: EXECUTE - Run tool calls generated in the decision stage
            jobs_to_execute = [j for j in jobs if j.get("current_step") == "tool_execution"]
            process_tool_execution_stage(jobs_to_execute, local_api, searxng_api)
            
            active_jobs = [j for j in jobs if j.get("current_step") not in ["evaluate", "done"]]
            if not active_jobs:
                print(f"All jobs completed or moved to evaluation by turn {turn + 1}. Exiting agent loop.")
                break
        
        for job in jobs:
            if job.get("current_step") != "evaluate":
                job["current_step"] = "evaluate"
        print("--- Agentic loop finished ---")

    # --- Final Evaluation and Saving ---

    # For ranking, we need an additional verdict generation step.
    if args.scheme == "ranking":
        print("--- Starting Final Evaluation Stage for Ranking ---")
        evaluation_jobs = [j for j in jobs if j.get("current_step") == "evaluate"]
        process_evaluation_stage(evaluation_jobs, engine, tokenizer, args)
        print("--- Final Evaluation Finished ---")
    
    # For pointwise, we skip the verdict step and proceed directly to saving.

    print("All sequences processed. Saving results...")
    final_results = []
    output_path = args.output_file

    # Prepare the final results list based on the scheme
    if args.scheme == "pointwise":
        for job in sorted(jobs, key=lambda x: x["id"]):
            result_item = job["original_item"]
            result_item["retrieval_path"] = job.get("messages", [])
            result_item["search_count"] = job.get("search_count", 0)
            final_results.append(result_item)
    else:  # ranking
        for job in sorted(jobs, key=lambda x: x["id"]):
            result_item = {
                **job["original_item"],
                "search_messages": job.get("messages", []),
                "verdict_response": job.get("verdict_response", ""),
                "predicted_ranking": job.get("predicted_ranking", []),
                "search_count": job.get("search_count", 0),
            }
            final_results.append(result_item)

    # Write the results to the output file
    with open(output_path, "w", encoding="utf-8") as f_out:
        for result in final_results:
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Results saved to {output_path}")

    # Perform final evaluation if applicable for the scheme
    if args.scheme == "ranking":
        evaluate_final_results_ranking(final_results)
    else: # pointwise
        print("Pointwise mode finished. Verdict step was skipped, so no evaluation metrics are calculated.")

    # Print final summary
    print(f"\nSummary:\nDataset: {os.path.basename(args.input_file)}\n"
          f"Model: {os.path.basename(args.model_path)}\n"
          f"Mode: {args.mode}\nScheme: {args.scheme}")


if __name__ == "__main__":
    main()