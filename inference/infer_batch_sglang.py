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

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
)


# --- Command-line Argument Parsing ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch inference script for fact-checking model using SGLang."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the SGLang-compatible model.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output JSONL file.",
    )
    parser.add_argument(
        "--gpu_idx",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["local_retrieval", "direct_gen"],
        help="Set the operating mode: 'local_retrieval' for search-based generation, 'direct_gen' for direct answering.",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="best_of_n",
        choices=["best_of_n", "pointwise"],
        help="Evaluation scheme. 'best_of_n' is for selecting the best answer from a list."
    )
    return parser.parse_args()


# --- Tool Definition & Implementation ---
SEARCH_TOOL_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "When you lack knowledge, you can call a search engine to get information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query keyword.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def search_api_call(queries: List[str]) -> List[str]:
    """
    The implementation of the search tool. Accepts a list of queries for batch processing.
    """
    if not queries:
        return []
    payload = {"queries": queries, "topk": 3, "return_scores": True}
    try:
        response = requests.post(
            "http://127.0.0.1:8000/retrieve", json=payload, timeout=2000
        )
        response.raise_for_status()
        results_list = response.json()["result"]
    except requests.RequestException as e:
        logging.error(f"Search API request failed for queries '{queries}': {e}")
        return ["Search failed due to connection error."] * len(queries)

    def _passages2string(retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item.get("document", {}).get("contents", "")
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return [_passages2string(results) for results in results_list]


# --- Result Evaluation Functions ---
def extract_final_verdict(model_generated_output: str) -> int:
    """
    Extracts the final verdict from the model's output, expecting formats like <verdict>Answer1</verdict>.
    Returns a zero-based index (e.g., "Answer1" -> 0).
    """
    # Updated pattern to capture the number from "AnswerX"
    answer_pattern = re.compile(r"<verdict>Answer(\d+)</verdict>", re.IGNORECASE | re.DOTALL)
    matches = answer_pattern.findall(model_generated_output)
    if matches:
        try:
            # Get the last match and convert it to a zero-based index
            answer_index = int(matches[-1])
            return answer_index - 1
        except (ValueError, IndexError):
            return -1 # Return -1 if parsing fails
    return -1 # Return -1 if no match is found


def check_ready_for_evaluation(model_generated_output: str) -> bool:
    """
    Checks if the model is ready for evaluation (i.e., no more searching needed).
    """
    return "READY_FOR_EVALUATION" in model_generated_output or "READY_FOR_ANSWERING" in model_generated_output


def evaluate_final_results(results: List[Dict]):
    """
    Calculates and prints evaluation metrics for a multi-class (best-of-N) scenario.
    """
    y_true, y_pred = [], []
    invalid_predictions = 0
    num_classes = 0
    if results:
        # Determine the number of classes from the first item's answers list
        num_classes = len(results[0].get("answers", []))

    for item in results:
        true_label = item.get("verify_result")
        pred_label = item.get("final_verdict")

        if true_label is None:
            continue # Skip items without a ground truth label

        # Handle invalid predictions
        if pred_label == -1 or pred_label >= num_classes:
            pred_label = 0  # Default to the first choice for metric calculation
            invalid_predictions += 1

        y_true.append(true_label)
        y_pred.append(pred_label)

    if not y_true:
        logging.error("Evaluation failed. No valid ground truth labels found.")
        return None

    # Use 'macro' average for multi-class precision, recall, and F1
    metrics_dict = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "evaluated_count": len(results),
        "invalid_predictions": invalid_predictions,
        "invalid_ratio": round(invalid_predictions / len(results) if results else 0, 4),
    }

    print("\n--- Evaluation Results ---")
    for key, value in metrics_dict.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("--------------------------\n")
    return metrics_dict

def batched_sglang_generation(input_ids, sampling_params, engine, BATCH_SIZE=100):
    """Generates text in batches using the SGLang engine."""
    batched_input_ids = [
        input_ids[i:i + BATCH_SIZE] 
        for i in range(0, len(input_ids), BATCH_SIZE)
    ]
    results = []
    for input_ids_batch in tqdm.tqdm(batched_input_ids, desc="Batched Generating"):
        results_batch = engine.generate(
            input_ids=input_ids_batch,
            sampling_params=sampling_params
        )
        results.extend(results_batch)
    return results

def process_search_stage(search_jobs, engine, tokenizer, parser, turn):
    """Processes jobs that require generating search queries (iterative, one query per turn)."""
    if not search_jobs:
        return

    print(f"--- Turn {turn + 1}: Processing {len(search_jobs)} search jobs ---")
    
    search_input_ids = [
        tokenizer.apply_chat_template(
            conversation=job["messages"],
            tokenize=True,
            add_generation_prompt=True,
            tools=SEARCH_TOOL_DEFINITION,
        )
        for job in search_jobs
    ]
    
    sampling_params = {"max_new_tokens": 1024, "temperature": 0}
    search_responses = batched_sglang_generation(
        input_ids=search_input_ids,
        sampling_params=sampling_params,
        engine=engine,
    )
    
    search_queries = []
    for job, response in zip(search_jobs, search_responses):
        job["search_query"] = None
        generated_text = response["text"]
        job["messages"].append({"role": "assistant", "content": generated_text})

        if check_ready_for_evaluation(generated_text):
            job["current_step"] = "evaluate"
            continue

        try:
            _, calls = parser.parse_non_stream(generated_text)
        except (json.JSONDecodeError, ValueError):
            logging.warning(f"Job {job['id']}: Failed to parse tool call. Moving to evaluate stage.")
            job["current_step"] = "evaluate"
            continue

        search_query = None
        if calls:
            for call in calls:
                if call.name == "search":
                    params = call.parameters
                    if isinstance(params, str):
                        try:
                            params = json.loads(params)
                        except json.JSONDecodeError:
                            logging.warning(f"Job {job['id']}: Failed to parse parameters JSON: {params}")
                            continue
                    
                    if isinstance(params, dict) and "query" in params:
                        search_query = params["query"]
                        search_queries.append(search_query)
                        job["search_query"] = search_query
                        break # Found one valid search query, process it and stop.

    api_results = search_api_call(search_queries)
    valid_search_job_ids = [i for i,job in enumerate(search_jobs) if job["search_query"] is not None]
    assert len(api_results) == len(search_queries)

    for api_result, id in zip(api_results, valid_search_job_ids):  
        search_jobs[id]["search_count"] += 1
        search_jobs[id]["messages"].append({"role": "tool", "content": api_result})
        search_jobs[id]["current_step"] = "search"
        search_jobs[id]["search_results"].append({"query": search_jobs[id]["search_query"], "result": api_result})
        del search_jobs[id]["search_query"]

def process_evaluation_stage(evaluate_jobs, engine, tokenizer, args):
    """Summarizes search results for jobs that performed searches."""

    def parse_info_tags(text: str) -> List[str]:
        """Extracts content from <Info>...</Info> tags using regex."""
        pattern = r"<Info>(.*?)</Info>"
        facts = re.findall(pattern, text, re.DOTALL)
        facts = [fact.strip() for fact in facts if fact.strip() != "No useful information retrieved."]
        return facts

    print(f"--- Processing {len(evaluate_jobs)} evaluation jobs ---")
    
    jobs_to_summarize = []
    for job in evaluate_jobs:
        if job["search_count"] == 0:
            job["current_step"] = "verdict"
            job["extracted_facts"] = []
        else:
            jobs_to_summarize.append(job)

    if not jobs_to_summarize:
        return
        
    print(f"--- Generating summaries for {len(jobs_to_summarize)} jobs ---")

    best_of_n_summarization_prompt = """Based on the preceding conversation, please provide a concise summary of the key facts you have gathered to help verify the factuality of the answer.
Please direct output the key facts with the format of '<Info> fact 1 </Info> <Info> fact 2 </Info> <Info> fact 3 </Info>', without any openings, closings or additional explanations.
If there is no useful information, you can direct output '<Info> No useful information retrieved. </Info>'""" 

    pointwise_summarization_prompt = """Based on the preceding conversation, please provide a concise summary of the key facts you have gathered to help answer the question.
Please direct output the key facts with the format of '<Info> fact 1 </Info> <Info> fact 2 </Info> <Info> fact 3 </Info>', without any openings, closings or additional explanations.
If there is no useful information, you can direct output '<Info> No useful information retrieved. </Info>'""" 

    if args.scheme == "best_of_n":
        summarization_prompt = best_of_n_summarization_prompt
    else:
        summarization_prompt = pointwise_summarization_prompt
    
    summary_input_ids = []
    for job in jobs_to_summarize:
        summary_messages = job["messages"] + [{"role": "user", "content": summarization_prompt}]
        summary_input_ids.append(tokenizer.apply_chat_template(
            conversation=summary_messages, tokenize=True, add_generation_prompt=True
        ))

    sampling_params = {"max_new_tokens": 1024, "temperature": 0}
    summary_responses = batched_sglang_generation(
        input_ids=summary_input_ids,
        sampling_params=sampling_params,
        engine=engine,
    )

    for job, response in zip(jobs_to_summarize, summary_responses):
        generated_summary = response["text"]
        extracted_facts = parse_info_tags(generated_summary)
        job["extracted_facts"] = extracted_facts
        job["current_step"] = "verdict"

def process_verdict_stage(verdict_jobs, engine, tokenizer, args):
    """Processes jobs that are ready for a final verdict."""
    if not verdict_jobs:
        return
    
    # Prompt for local_retrieval mode
    retrieval_verdict_prompt = """You are an expert fact-checking assistant. Your task is to determine which answer is the most factually correct to the question among the given options.

Question: {question}
{answers_block}
Retrieved Reference Information: {search_summary}

Based on the question, all the provided answers, and the reference information, determine which answer is the most factually correct answer to the question.
Please provide your explanation first, and then state your final verdict in the format: 'Therefore, the best answer is: <verdict>AnswerX</verdict>', where X is the number of the best answer. For example, 'Therefore, the best answer is: <verdict>Answer3</verdict>'"""

    # Prompt for direct_gen mode
    direct_gen_verdict_prompt = """You are an expert fact-checking assistant. Your task is to determine which answer is the most factually correct to the question among the given options.

Question: {question}
{answers_block}

Based on the question and the provided answers, determine which answer is the most factually correct answer to the question.
Please provide your explanation first, and then state your final verdict in the format: 'Therefore, the best answer is: <verdict>AnswerX</verdict>'. For example, 'Therefore, the best answer is: <verdict>Answer3</verdict>'"""

    print(f"--- Processing {len(verdict_jobs)} verdict jobs ---")
    verdict_input_ids = []
    for job in verdict_jobs:
        # Dynamically create the block of answers
        answers = job["original_item"]["answers"]
        answers_block = "\n".join([f"Answer{i+1}: {ans['answer']}" for i, ans in enumerate(answers)])

        if args.mode == "local_retrieval":
            facts = job.get("extracted_facts", [])
            if not facts:
                formatted_facts = "No useful information retrieved."
            else:
                formatted_facts = " ".join([f"{i+1}. {fact}" for i, fact in enumerate(facts)])
            
            prompt = retrieval_verdict_prompt.format(
                question=job["original_item"]["question"],
                answers_block=answers_block,
                search_summary=formatted_facts
            )
        else:  # direct_gen mode
            prompt = direct_gen_verdict_prompt.format(
                question=job["original_item"]["question"],
                answers_block=answers_block
            )
        
        verdict_messages = [{"role": "user", "content": prompt}]
        verdict_input_ids.append(tokenizer.apply_chat_template(
            conversation=verdict_messages, tokenize=True, add_generation_prompt=True
        ))

    sampling_params = {"max_new_tokens": 1024, "temperature": 0}
    verdict_responses = batched_sglang_generation(
        input_ids=verdict_input_ids,
        sampling_params=sampling_params,
        engine=engine,
    )
    
    for job, response in zip(verdict_jobs, verdict_responses):
        job["verdict_response"] = response["text"]

# --- Main Function ---
def main():
    """Main execution function, refactored for SGLang."""
    args = parse_args()
    MAX_TURNS = 5 

    print(f"Arguments: {args}")

    print("Initializing SGLang engine...")
    engine = sgl.Engine(model_path=args.model_path, trust_remote_code=True, mem_fraction_static=0.8)
    tokenizer = engine.tokenizer_manager.tokenizer

    def convert_dict_to_tool(tool_dict: dict) -> Tool:
        function_dict = tool_dict.get("function", {})
        return Tool(type=tool_dict.get("type", "function"), function=Function(**function_dict))
    tools = [convert_dict_to_tool(raw_tool) for raw_tool in SEARCH_TOOL_DEFINITION]
    parser = FunctionCallParser(tools=tools, tool_call_parser="qwen25")

    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f if line.strip()]

    # --- Prompt Templates ---
    best_of_n_search_prompt = f"""You are an expert fact-checking assistant. Your task is to determine which of the provided answers is the most factually correct.

Based on the question and the answers below, and any existing search results, identify what additional knowledge you need to compare their factual accuracy, and generate search queries for the missing information.

Question: {{question}}
{{answers_block}}

Please analyze what information is still needed and generate search queries one by one using <tool_call> query </tool_call>. The search results will be returned between <tool_response> and </tool_response>.

If you believe you have enough information to make a judgment and don't need to search any more, respond with "READY_FOR_EVALUATION" instead of making tool calls."""

    pointwise_search_prompt = f"""Your task is to gather information and answer the following question. First identify what knowledge you need to answer the question, and generate search queries for the missing information.

Question: {{question}}

Please analyze what information is needed and generate search queries one by one using <tool_call> query </tool_call>. The search results will be returned between <tool_response> and </tool_response>.

If you believe you have enough information to answer the question and don't need to search any more, respond with "READY_FOR_ANSWERING" instead of making tool calls."""

    # --- Initialize Job Queue ---
    jobs = []
    for i, item in enumerate(input_data):
        if args.scheme == "pointwise":
            # MODIFIED: Create one search job per question, not per answer.
            content = pointwise_search_prompt.format(
                question=item["question"]
            )
            jobs.append({
                "id": i,
                "original_item": item,
                "search_count": 0,
                "messages": [{"role": "user", "content": content}],
                "current_step": "search",
                "search_results": []
            })
        else: # best_of_n
            job_base = {"id": i, "original_item": item, "search_count": 0}
            if args.mode == "local_retrieval":
                answers = item.get("answers", [])
                answers_block = "\n".join([f"Answer{i+1}: {ans['answer']}" for i, ans in enumerate(answers)])
                content = best_of_n_search_prompt.format(
                    question=item["question"], 
                    answers_block=answers_block,
                )
                jobs.append({**job_base, "messages": [{"role": "user", "content": content}], "current_step": "search", "search_results": []})
            elif args.mode == "direct_gen":
                jobs.append({**job_base, "current_step": "verdict", "extracted_facts": []})

    # --- Conditional Execution based on Mode ---
    if args.mode == "local_retrieval":
        # Stage 1: Iterative Search
        print(f"--- Starting Iterative Search Stage ---")
        for turn in tqdm.tqdm(range(MAX_TURNS), desc="Searching"):
            search_jobs = [job for job in jobs if job["current_step"] == "search"]
            if not search_jobs:
                print("No more active jobs in search stage. Exiting search loop.")
                break
            process_search_stage(search_jobs, engine, tokenizer, parser, turn)
        print("--- Search Stage Finished ---")

        # Stage 2: Evaluation (Summarization)
        print("--- Starting Evaluation Stage ---")
        for job in jobs:
            job["current_step"] = "evaluate"
        evaluation_jobs = [job for job in jobs if job["current_step"] == "evaluate"]
        process_evaluation_stage(evaluation_jobs, engine, tokenizer, args)
        print("--- Evaluation Stage Finished ---")
    
    # --- Pointwise scheme skips verdict and final evaluation ---
    if args.scheme == "pointwise":
        print("Pointwise scheme selected. Skipping verdict and final evaluation.")
        with open(args.output_file, "w", encoding="utf-8") as f_out:
            # MODIFIED: Process results for one-job-per-question logic.
            for job in sorted(jobs, key=lambda x: x["id"]):
                # Start with the original item from the job
                result_item = job["original_item"]
                
                # Get the single set of extracted facts
                extracted_facts = job.get("extracted_facts", [])
                
                # Create a new list of answers, adding the same facts to each one
                updated_answers = []
                for ans in result_item.get("answers", []):
                    # Copy original answer and add the extracted_facts key
                    updated_answers.append({**ans, "extracted_facts": extracted_facts})
                
                # Replace the original answers list with the updated one
                result_item["answers"] = updated_answers
                
                # Write the final combined result to the output file
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
        
        print(f"Results saved to {args.output_file}")
        return # End execution for pointwise

    # --- Best-of-N final stages ---
    # Stage 3: Verdict (runs for best_of_n mode)
    print("--- Starting Verdict Stage ---")
    verdict_jobs = [job for job in jobs if job["current_step"] == "verdict"]
    process_verdict_stage(verdict_jobs, engine, tokenizer, args)
    print("--- Verdict Stage Finished ---")
    
    # --- Result Processing and Saving ---
    print("All sequences processed. Saving results...")
    final_results = []
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for job in sorted(jobs, key=lambda x: x["id"]):
            final_verdict = extract_final_verdict(job.get("verdict_response", ""))
            result_item = {
                **job["original_item"],
                "search_messages": job.get("messages", []),
                "extracted_facts": job.get("extracted_facts", []),
                "verdict_response": job.get("verdict_response", ""),
                "final_verdict": final_verdict,
                "search_count": job.get("search_count", 0),
            }
            final_results.append(result_item)
            f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")

    print(f"Results saved to {args.output_file}")
    
    input_file_name = os.path.basename(args.input_file)
    model_name = os.path.basename(args.model_path)
    print(f"\nDataset: {input_file_name}\nModel: {model_name}\nMode: {args.mode}")
    
    print("Starting evaluation...")
    metrics = evaluate_final_results(final_results)
    if metrics:
        print(f"Evaluation metrics: {json.dumps(metrics, indent=4)}")


if __name__ == "__main__":
    main()