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

from search_api_local import SearchAPILocal
from search_api_serper import SearchAPISerper
from search_api_searxng import SearchAPISearxng


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
    parser.add_argument(
        "--search_api",
        type=str,
        choices=["local", "serper", "searxng"],
        default="local"
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        default=False,
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


# def search_api_call(queries: List[str]) -> List[str]:
#     """
#     The implementation of the search tool. Accepts a list of queries for batch processing.
#     """
#     if not queries:
#         return []
#     payload = {"queries": queries, "topk": 3, "return_scores": True}
#     try:
#         response = requests.post(
#             "http://127.0.0.1:8000/retrieve", json=payload, timeout=2000
#         )
#         response.raise_for_status()
#         results_list = response.json()["result"]
#     except requests.RequestException as e:
#         logging.error(f"Search API request failed for queries '{queries}': {e}")
#         return ["Search failed due to connection error."] * len(queries)

#     def _passages2string(retrieval_result):
#         format_reference = ""
#         for idx, doc_item in enumerate(retrieval_result):
#             content = doc_item.get("document", {}).get("contents", "")
#             title = content.split("\n")[0]
#             text = "\n".join(content.split("\n")[1:])
#             format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
#         return format_reference

#     return [_passages2string(results) for results in results_list]


def extract_final_verdict_best_of_n(model_generated_output: str) -> int:
    """
    Extracts the final verdict from the model's output. It first looks for formats
    like <verdict>Answer1</verdict>, and if not found, falls back to searching for
    **Final Verdict**: Answer1.

    Returns a zero-based index (e.g., "Answer1" -> 0).
    """
    if not isinstance(model_generated_output, str):
        return -1 # Gracefully handle non-string input

    matches = []
    
    # Pattern 1: <verdict>AnswerX</verdict>
    # Added \s* to be more robust against whitespace
    answer_pattern_1 = re.compile(r"<verdict>\s*Answer(\d+)\s*</verdict>", re.IGNORECASE | re.DOTALL)
    matches = answer_pattern_1.findall(model_generated_output)
    
    # If Pattern 1 fails, try Pattern 2: **Final Verdict**: AnswerX
    if not matches:
        answer_pattern_2 = re.compile(r"\*\*Final Verdict\*\*:\s*Answer(\d+)", re.IGNORECASE)
        matches = answer_pattern_2.findall(model_generated_output)
        
    if matches:
        try:
            # Get the last match and convert it to a zero-based index
            answer_index = int(matches[-1])
            return answer_index - 1
        except (ValueError, IndexError):
            return -1 # Return -1 if parsing fails
            
    return -1 # Return -1 if no match is found

def extract_final_verdict_pointwise(model_generated_output: str) -> str:
    """
    Extracts the final textual verdict from the model's output. It first looks for
    formats like <verdict>Correct</verdict>, and if not found, falls back to
    searching for **Final Verdict**: Correct.

    Returns the extracted text (e.g., "Correct", "Incorrect") as a string, or
    "Invalid" if no match is found.
    """
    if not isinstance(model_generated_output, str):
        return None  # Gracefully handle non-string input

    matches = []

    # Pattern 1: <verdict>...</verdict>
    verdict_pattern_1 = re.compile(r"<verdict>(.*?)</verdict>", re.IGNORECASE | re.DOTALL)
    matches = verdict_pattern_1.findall(model_generated_output)
    
    # If Pattern 1 fails, try Pattern 2: **Final Verdict**: ...
    if not matches:
        # This pattern captures everything after "**Final Verdict**:"
        verdict_pattern_2 = re.compile(r"\*\*Final Verdict\*\*:\s*(.*)", re.IGNORECASE | re.DOTALL)
        matches = verdict_pattern_2.findall(model_generated_output)
    
    if matches:
        # If one or more matches are found, return the last one, stripped of whitespace.
        return matches[-1].strip()
    
    # If no valid pattern is found in the output, return "Invalid"
    return "Invalid"

def check_ready_for_evaluation(model_generated_output: str) -> bool:
    """
    Checks if the model is ready for evaluation (i.e., no more searching needed).
    """
    return "READY_FOR_EVALUATION" in model_generated_output or "READY_FOR_ANSWERING" in model_generated_output


def evaluate_final_results_best_of_n(results: List[Dict]):
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
        "precision": round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "invalid_ratio": round(invalid_predictions / len(results) if results else 0, 4),
    }

    print("\n--- Evaluation Results ---")
    for key, value in metrics_dict.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("--------------------------\n")
    return metrics_dict


def evaluate_final_results_pointwise(results: List[Dict]):
    """
    为逐点（Pointwise）评估场景计算并打印二元分类指标。

    该函数会遍历每个样本中的每一个答案，将其视为一个独立的二元分类问题：
    这个答案是否正确？
    """
    tp, tn, fp, fn = 0, 0, 0, 0
    total_answers = 0
    invalid_predictions = 0

    for item in results:
        # 假设 `results` 列表中的每个 item 都包含一个名为 "answers" 的列表
        answers = item.get("answers", [])
        if not answers:
            continue

        for answer in answers:
            total_answers += 1
            
            ground_truth_str = answer.get("verify_result", "").lower()
            final_verdict_str = answer.get("final_verdict", "").lower()

            if final_verdict_str == "invalid":
                invalid_predictions += 1
            
            # 定义正负例的判断标准
            # 基准真相 (Ground truth) 为正例，如果它是 "correct" 或 "intermediate"
            is_ground_truth_positive = ground_truth_str in ["correct", "intermediate"]
            is_prediction_positive = final_verdict_str in ["correct", "intermediate"]

            # 根据判断更新混淆矩阵的计数
            if is_ground_truth_positive and is_prediction_positive:
                tp += 1
            elif not is_ground_truth_positive and not is_prediction_positive:
                tn += 1
            elif not is_ground_truth_positive and is_prediction_positive:
                fp += 1
            elif is_ground_truth_positive and not is_prediction_positive:
                fn += 1

    # 基于有效预测计算各项指标
    accuracy = (tp + tn) / total_answers if total_answers > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    invalid_ratio = invalid_predictions / total_answers if total_answers > 0 else 0
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "invalid_ratio": invalid_ratio,
    }
    
    print("\n--- Pointwise Evaluation Metrics ---")
    for key, value in metrics.items():
        # 对浮点数进行格式化，保留4位小数
        formatted_value = f"{value:.4f}" if isinstance(value, float) else value
        print(f"{key.replace('_', ' ').title()}: {formatted_value}")
    print("------------------------------------")
    
    return metrics


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

def process_search_stage(search_jobs, search_api, engine, tokenizer, parser, turn, args):
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
            enable_thinking=(not args.disable_thinking),
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
    
    api_results = search_api.search_api_call(search_queries)
    valid_search_job_ids = [i for i,job in enumerate(search_jobs) if job["search_query"] is not None]
    assert len(api_results) == len(search_queries)

    for api_result, id in zip(api_results, valid_search_job_ids):  
        search_jobs[id]["search_count"] += 1
        search_jobs[id]["messages"].append({"role": "tool", "content": api_result})
        search_jobs[id]["current_step"] = "search"
        search_jobs[id]["search_results"].append({"query": search_jobs[id]["search_query"], "result": api_result})
        del search_jobs[id]["search_query"]

def process_evaluation_stage(evaluate_jobs, engine, tokenizer, args):
    """
    Processes jobs that are ready for evaluation and final verdict.
    This function combines summarization and verdict generation into a single step.
    """
    if not evaluate_jobs:
        return

    def parse_info_tags(text: str) -> List[str]:
        pattern = r"<facts>(.*?)</facts>"
        facts = re.findall(pattern, text, re.DOTALL)
        if facts == []:
            pattern = r"\*\*Useful Information\*\*: (.*?)\*\*Verdict Reasoning\*\*"
            facts = re.findall(pattern, text, re.DOTALL)     
            if facts == []:
                return ["No useful information retrieved."]
            else:
                return [fact.strip() for fact in facts]              
        return [fact.strip() for fact in facts]
    
    def parse_reasoning_tags(text: str)-> str:
        pattern = r"<reasoning>(.*?)</reasoning>"
        reasoning = re.findall(pattern, text, re.DOTALL)
        if reasoning == []:
            pattern = r"\*\*Verdict Reasoning\*\*: (.*?)\*\*Final Verdict\*\*"
            reasoning = re.findall(pattern, text, re.DOTALL)     
            if reasoning == []:
                return "No reasoning process detected."
            else:
                return reasoning[0].strip()                
        return reasoning[0].strip()

    print(f"--- Processing {len(evaluate_jobs)} jobs for evaluation and verdict ---")
    
    sampling_params = {"max_new_tokens": 1024, "temperature": 0}

    if args.scheme == "best_of_n":

        best_of_n_verdict_prompt = """Based on the preceding conversation, your task is to first summarize the information you have retrieved, and then determine which answer is the most factually correct for the question.

Question: {question}
{answers_block}

Please perform the following three steps:
1. Summarize the useful information from the retrieval results in the format: '**Useful Information**: <facts> Fact 1; Fact 2; Fact 3 </facts>'. If no useful information was found, use '**Useful Information**: <facts> No useful information retrieved. </facts>'.
2. Based on summarized information, provide your reasoning for the verdict in the format: '**Verdict Reasoning**: <reasoning> Your reasoning process... </reasoning>'.
3. After reasoning, state your final verdict in the format: '**Final Verdict**: <verdict> AnswerX </verdict>'. For example, '**Final Verdict**: <verdict> Answer3 </verdict>.'"""

        direct_gen_verdict_prompt = """You are an expert fact-checking assistant. Your task is to determine which answer is the most factually correct to the question among the given options.

Question: {question}
{answers_block}

Based on the question and the provided answers, determine which answer is the most factually correct answer to the question.
Please provide your explanation first, and then state your final verdict in the format: '**Final Verdict**: <verdict> AnswerX </verdict>'. For example, '**Final Verdict**: <verdict> Answer3 </verdict>.'"""

        verdict_input_ids = []
        for job in evaluate_jobs:
            answers = job["original_item"]["answers"]
            answers_block = "\n".join([f"Answer{i+1}: {ans['answer']}" for i, ans in enumerate(answers)])
            
            if args.mode == "local_retrieval":
                prompt = best_of_n_verdict_prompt.format(
                    question=job["original_item"]["question"],
                    answers_block=answers_block
                )
                final_messages = job["messages"] + [{"role": "user", "content": prompt}]
            else: # direct_gen mode
                prompt = direct_gen_verdict_prompt.format(
                    question=job["original_item"]["question"],
                    answers_block=answers_block
                )
                final_messages = [{"role": "user", "content": prompt}]
            
            verdict_input_ids.append(tokenizer.apply_chat_template(
                conversation=final_messages, tokenize=True, add_generation_prompt=True, enable_thinking=(not args.disable_thinking)
            ))

        verdict_responses = batched_sglang_generation(
            input_ids=verdict_input_ids,
            sampling_params=sampling_params,
            engine=engine,
        )
        
        for job, response in zip(evaluate_jobs, verdict_responses):
            generated_text = response["text"]
            job["verdict_response"] = generated_text

            # Parse both facts and verdict from the single response
            job["extracted_facts"] = parse_info_tags(generated_text)
            job["reasoning_content"] = parse_reasoning_tags(generated_text)
            job["final_verdict"] = extract_final_verdict_best_of_n(generated_text)

    else: # pointwise scheme
        from utils_prompts import POINTWISE_VERDICT_PROMPT
        pointwise_verdict_prompt = POINTWISE_VERDICT_PROMPT
        
        pointwise_input_ids = []
        response_map = [] # Map each prompt back to its original job and answer
        for job in evaluate_jobs:
            question = job["original_item"]["question"]
            for i, ans in enumerate(job["original_item"]["answers"]):
                prompt_content = pointwise_verdict_prompt.format(
                    question=question,
                    answer=ans['answer']
                )
                # For pointwise, which is always retrieval, we use the message history
                final_messages = job["messages"] + [{"role": "user", "content": prompt_content}]
                
                tokenized_prompt = tokenizer.apply_chat_template(
                    conversation=final_messages, tokenize=True, add_generation_prompt=True, enable_thinking=(not args.disable_thinking)
                )
                pointwise_input_ids.append(tokenized_prompt)
                response_map.append({"job": job, "answer_index": i})
        
        if not pointwise_input_ids:
            print("No answers found to generate pointwise verdicts for.")
            return

        print(f"--- Generating combined summary and verdicts for {len(pointwise_input_ids)} individual answers ---")
        verdict_responses = batched_sglang_generation(
            input_ids=pointwise_input_ids,
            sampling_params=sampling_params,
            engine=engine,
        )
        
        # Map responses back to their corresponding answers
        for i, response in enumerate(verdict_responses):
            mapping = response_map[i]
            job = mapping["job"]
            answer_index = mapping["answer_index"]
            generated_text = response["text"]
            
            # Store the full response, extracted facts, and extracted verdict
            answer = job["original_item"]["answers"][answer_index]
            answer["verdict_response"] = generated_text

            answer["reasoning_content"] = parse_reasoning_tags(generated_text)
            answer["extracted_facts"] = parse_info_tags(generated_text)
            answer["final_verdict"] = extract_final_verdict_pointwise(generated_text)
            answer["retrieval_path"] = job["messages"]


# --- Main Function ---
def main():
    """Main execution function, refactored for SGLang."""
    args = parse_args()
    MAX_TURNS = 5

    # Add validation for pointwise mode
    if args.scheme == "pointwise" and args.mode == "direct_gen":
        print("Error: 'direct_gen' mode is not supported for the 'pointwise' scheme. 'pointwise' always requires 'local_retrieval'.")
        exit(1)

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
            # For pointwise, always set up for local_retrieval
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
                # For direct_gen, it will skip search and go straight to evaluation
                jobs.append({**job_base, "current_step": "evaluate"})

    # --- Conditional Execution based on Mode ---
    if args.mode == "local_retrieval":
        
        if args.search_api == "local":
            search_api = SearchAPILocal()
        elif args.search_api == "serper":
            search_api = SearchAPISerper()
        elif args.search_api == "searxng":
            search_api = SearchAPISearxng()

        # Stage 1: Iterative Search
        print(f"--- Starting Iterative Search Stage ---")
        for turn in tqdm.tqdm(range(MAX_TURNS), desc="Searching"):
            search_jobs = [job for job in jobs if job["current_step"] == "search"]
            if not search_jobs:
                print("No more active jobs in search stage. Exiting search loop.")
                break
            process_search_stage(search_jobs, search_api, engine, tokenizer, parser, turn, args)
        print("--- Search Stage Finished ---")
        # Mark all remaining search jobs as ready for evaluation
        for job in jobs:
            if job["current_step"] == "search":
                job["current_step"] = "evaluate"

    # Stage 2: Combined Evaluation and Verdict Stage
    print("--- Starting Combined Evaluation and Verdict Stage ---")
    evaluation_jobs = [job for job in jobs if job["current_step"] == "evaluate"]
    process_evaluation_stage(evaluation_jobs, engine, tokenizer, args)
    print("--- Combined Stage Finished ---")
    
    # --- Result Processing and Saving ---
    print("All sequences processed. Saving results...")

    if args.scheme == "pointwise":
        final_results = []
        with open(args.output_file, "w", encoding="utf-8") as f_out:
            for job in sorted(jobs, key=lambda x: x["id"]):
                # In pointwise, the verdicts and facts are already in original_item's answers
                result_item = job["original_item"]
                final_results.append(result_item)
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
        
        print(f"Results saved to {args.output_file}")
        input_file_name = os.path.basename(args.input_file)
        model_name = os.path.basename(args.model_path)
        print(f"\nDataset: {input_file_name}\nModel: {model_name}\nMode: {args.mode}")
        
        print("Starting evaluation...")
        metrics = evaluate_final_results_pointwise(final_results)
        if metrics:
            print(f"Evaluation metrics: {json.dumps(metrics, indent=4)}")
    
    else: # best_of_n
        final_results = []
        with open(args.output_file, "w", encoding="utf-8") as f_out:
            for job in sorted(jobs, key=lambda x: x["id"]):
                result_item = {
                    **job["original_item"],
                    "search_messages": job.get("messages", []),
                    "extracted_facts": job.get("extracted_facts", []),
                    "reasoning_content": job.get("reasoning_content", ""),
                    "verdict_response": job.get("verdict_response", ""),
                    "final_verdict": job.get("final_verdict", -1),
                    "search_count": job.get("search_count", 0),
                }
                final_results.append(result_item)
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")

        print(f"Results saved to {args.output_file}")
        
        input_file_name = os.path.basename(args.input_file)
        model_name = os.path.basename(args.model_path)
        print(f"\nDataset: {input_file_name}\nModel: {model_name}\nMode: {args.mode}")
        
        print("Starting evaluation...")
        metrics = evaluate_final_results_best_of_n(final_results)
        if metrics:
            print(f"Evaluation metrics: {json.dumps(metrics, indent=4)}")


if __name__ == "__main__":
    main()
