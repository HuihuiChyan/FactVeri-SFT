import argparse
import json
import logging
import re
import requests
from typing import List, Dict
import os
import asyncio
import tqdm
import sys
import time
import multiprocessing
import openai
from functools import partial
from multiprocessing import Pool

# Import metrics libraries
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau

# Import locally defined search API modules
from search_api_local import SearchAPILocal
# from search_api_searxng import SearchAPISearxng
from search_api_serper import SearchAPISerper as SearchAPISearxng

# --- Tool Definition ---
# Define the tools for the OpenAI API in the required format
SEARCH_TOOL_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "search_local",
            "description": "Searches the local Wikipedia document database.",
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
            "description": "Performs a web search using Google.",
            "parameters": {
                "type": "object",
                "properties": { "query": { "type": "string", "description": "The search query keyword for the web search engine." } },
                "required": ["query"],
            },
        },
    }
]


# --- Globals for API request counter ---
start_time = time.time()
counter = multiprocessing.Value('i', 0)

# Initialization function for multiprocessing.Pool to pass shared variables
def init_worker(c, t):
    global counter
    global start_time
    counter = c
    start_time = t

# --- OpenAI API Request Function ---
def request_gpt(messages: list, model: str, temperature: float, tools: list = None) -> openai.types.chat.ChatCompletionMessage:
    """
    Makes a request to the OpenAI-compatible API.
    Uses environment variables for credentials, with fallbacks.
    Returns the full ChatCompletionMessage object.
    """
    # Use env variables first, fallback to hardcoded values
    api_key = os.getenv("OPENAI_API_KEY", "sk-agEcX3Su78Bu09c2F49978C6Ba424977B936C8710fAb42E0")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.shubiaobiao.cn/v1/")

    if not api_key:
        raise ValueError("API key not found. Please set OPENAI_API_KEY or check the script.")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    create_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    
    if tools:
        create_params["tools"] = tools
        create_params["tool_choice"] = "auto"

    max_tries = 3
    res_message = None
    for i in range(max_tries):
        try:
            chat_completion = client.chat.completions.create(**create_params)
            res_message = chat_completion.choices[0].message
            break  # Exit loop on success
        except Exception as e:
            print(f"API Error (Attempt {i+1}/{max_tries}): {e}", file=sys.stderr)
            time.sleep(5)
            continue

    # --- Original counter logic from user's function ---
    with counter.get_lock():
        counter.value += 1
        count_val = counter.value
    
    if count_val % 10 == 0: # Log every 10 calls
        avg_time = round((time.time() - start_time) / count_val, 2)
        print(f"API call {count_val} finished! {avg_time}s avg per call.", file=sys.stderr)
    # --- End of counter logic ---

    if not res_message:
        print(f"API call failed after {max_tries} tries.", file=sys.stderr)
        # Return a dummy message object to avoid crashing the loop
        return openai.types.chat.ChatCompletionMessage(role="assistant", content="API Call Failed")

    return res_message

# --- Command-line Argument Parsing ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch inference script for fact-checking model using OpenAI."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the OpenAI model to use (e.g., gpt-4-turbo).",
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to save the output JSONL file.",
    )
    parser.add_argument(
        "--mode", type=str, required=True, choices=["retrieval", "direct_gen"],
        help="Set operating mode: 'retrieval' (search-based) or 'direct_gen' (direct generation)."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature for generation."
    )
    parser.add_argument(
        "--disable_cache_for_serper", action="store_true", default=False,
    )
    parser.add_argument(
        "--multi_process", action="store_true", default=False,
        help="Enable multiprocessing for API calls."
    )
    return parser.parse_args()

def extract_final_ranking(model_generated_output: str) -> List[int]:
    """
    Extracts the final ranked list from the model's output.
    
    It first tries to find content within <verdict>...</verdict> tags.
    If not found, it looks for the pattern AnswerX > AnswerY > AnswerZ.
    
    In both cases, it uses the *last* occurrence found.
    """
    if not isinstance(model_generated_output, str):
        return []

    # 1. 尝试匹配 <verdict>(.*?)</verdict>
    verdict_matches = re.findall(r"<verdict>(.*?)</verdict>", model_generated_output, re.IGNORECASE | re.DOTALL)
    
    content_to_parse = ""
    if verdict_matches:
        # 如果找到，使用最后一个匹配项的内容
        content_to_parse = verdict_matches[-1].strip()
    else:
        # 2. 如果没找到 <verdict>，尝试匹配 Answer1 > Answer2 > ... 模式
        #    使用 re.findall 找到所有匹配，并使用最后一个
        ranking_matches = re.findall(r'(Answer\d+\s*(?:>\s*Answer\d+\s*)*)', model_generated_output, re.IGNORECASE | re.DOTALL)
        
        if ranking_matches:
            # 使用最后一个匹配的排名字符串
            content_to_parse = ranking_matches[-1].strip()
        else:
            return [] # 两种模式都未匹配到

    try:
        # 3. 从选定的内容中提取所有 Answer 编号
        #    这个 findall 确保我们能获取到 "1", "2", "3"
        answer_numbers = re.findall(r'Answer(\d+)', content_to_parse, re.IGNORECASE)
        
        # 将捕获到的字符串数字转换为整数列表
        return [int(num) for num in answer_numbers]
        
    except ValueError:
        # 如果 int() 转换失败，返回空列表
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


# --- Agent State Processing Functions ---

def process_decision_stage(jobs_to_decide, model_name, args):
    """
    Decides the next action (search or evaluate) based on the conversation history.
    Uses native OpenAI tool calling.
    """
    if not jobs_to_decide: return
    
    print(f"--- Deciding next action for {len(jobs_to_decide)} jobs ---")
    
    # Prepare inputs for all jobs
    inputs = [job["messages"] for job in jobs_to_decide]
    responses = []

    if args.multi_process:
        print(f"Using multiprocessing for {len(inputs)} decision jobs...")
        # Pass the tools definition to the partial function
        pool_fn = partial(request_gpt, model=model_name, temperature=args.temperature, tools=SEARCH_TOOL_DEFINITION)
        with Pool(initializer=init_worker, initargs=(counter, start_time)) as pool:
            responses = list(tqdm.tqdm(pool.imap(pool_fn, inputs), total=len(inputs), desc="Deciding (Parallel)"))
    else:
        print("Processing decisions sequentially...")
        for messages in tqdm.tqdm(inputs, desc="Deciding (Sequential)"):
            response = request_gpt(
                messages=messages, 
                model=model_name, 
                temperature=args.temperature,
                tools=SEARCH_TOOL_DEFINITION
            )
            responses.append(response)

    # Process responses (which are now ChatCompletionMessage objects)
    for job, response_message in zip(jobs_to_decide, responses):
        # Append the full response message (as a dict) to history
        job["messages"].append(response_message.to_dict())
        
        generated_text = response_message.content or "" # Content can be None if only tool_calls

        # Check for tool calls
        if response_message.tool_calls:
            # Store tool_calls as dicts for JSON serialization
            job["tool_calls"] = [tc.to_dict() for tc in response_message.tool_calls]
            job["current_step"] = "tool_execution"
            continue

        # Check for ready signal
        if check_ready_for_evaluation(generated_text):
            job["current_step"] = "evaluate"
            continue
        
        # Fallback: if no tool call and no ready signal, assume it's ready for evaluation
        job["current_step"] = "evaluate"

def process_tool_execution_stage(jobs_with_calls, local_api, searxng_api):
    """Executes the tool calls, adds result to history, and sets state to decision."""
    if not jobs_with_calls: return
    print(f"--- Executing tools for {len(jobs_with_calls)} jobs ---")

    local_tasks, web_tasks = [], []
    for job in jobs_with_calls:
        if not job.get("tool_calls"): continue
        
        # --- FIX: Iterate over ALL tool calls, not just the first one ---
        for call in job["tool_calls"]: # 'call' is a dict
            try:
                call_id = call['id']
                call_name = call['function']['name']
                params = json.loads(call['function']['arguments'])
                query = params.get("query")

                if not query: continue
                
                if call_name == "search_local": 
                    local_tasks.append({"job": job, "query": query, "tool_call_id": call_id})
                elif call_name == "search_web": 
                    web_tasks.append({"job": job, "query": query, "tool_call_id": call_id})
            
            except Exception as e:
                logging.warning(f"Job {job['id']}: Failed to parse tool call ({call.get('function', {}).get('name', 'N/A')}): {e}")
                job["current_step"] = "decision" # Go back to decision if tool call fails
        # --- END FIX ---

    if local_tasks:
        results = local_api.search_api_call([t["query"] for t in local_tasks])
        for task, result in zip(local_tasks, results):
            content = f"[Source: Local Wikipedia]\n{result}"
            # Append the tool result with the corresponding tool_call_id
            task["job"]["messages"].append({
                "role": "tool", 
                "content": content, 
                "tool_call_id": task["tool_call_id"]
            })
            task["job"]["current_step"] = "decision"
            task["job"]["search_count"] += 1

    if web_tasks:
        results = searxng_api.search_api_call([t["query"] for t in web_tasks])
        for task, result in zip(web_tasks, results):
            content = f"[Source: Google Search]\n{result}"
            # Append the tool result with the corresponding tool_call_id
            task["job"]["messages"].append({
                "role": "tool", 
                "content": content, 
                "tool_call_id": task["tool_call_id"]
            })
            task["job"]["current_step"] = "decision"
            task["job"]["search_count"] += 1
            
    # Clean up tool_calls after execution
    for job in jobs_with_calls:
        if "tool_calls" in job:
            del job["tool_calls"]


def process_evaluation_stage_ranking(evaluate_jobs, model_name, args):
    """Processes jobs ready for final evaluation and verdict."""
    if not evaluate_jobs: return
    
    print(f"--- Starting final evaluation for {len(evaluate_jobs)} jobs ---")

    ranking_verdict_prompt_retrieval = """Based on the preceding conversation, your task is to rank all the given answers from most to least factually correct.

Question: {question}
{answers_block}

Please first provide your explanation, and then state your final verdict in the format: '**Final Verdict**: <verdict> AnswerX > AnswerY > AnswerZ </verdict>'. Example: '**Final Verdict**: <verdict> Answer3 > Answer1 > Answer2 </verdict>'."""

    ranking_verdict_prompt_direct = """You are an expert fact-checking assistant. Your task is to rank all the given answers from most to least factually correct based on your internal knowledge.

Question: {question}
{answers_block}

Please first provide your explanation, and then state your final verdict in the format: '**Final Verdict**: <verdict> AnswerX > AnswerY > AnswerZ </verdict>'. Example: '**Final Verdict**: <verdict> Answer3 > Answer1 > Answer2 </verdict>'."""

    inputs = []
    for job in evaluate_jobs:
        answers = job["original_item"]["answers"]
        answers_block = "\n".join([f"Answer{i+1}: {ans['answer']}" for i, ans in enumerate(answers)])
        
        if args.mode == "retrieval":
            prompt = ranking_verdict_prompt_retrieval.format(question=job["original_item"]["question"], answers_block=answers_block)
            final_messages = job["messages"] + [{"role": "user", "content": prompt}]
        else: # direct_gen mode
            prompt = ranking_verdict_prompt_direct.format(question=job["original_item"]["question"], answers_block=answers_block)
            final_messages = [{"role": "user", "content": prompt}]
        inputs.append(final_messages)

    responses = []
    if args.multi_process:
        print(f"Using multiprocessing for {len(inputs)} evaluation jobs...")
        # No tools needed for final verdict
        pool_fn = partial(request_gpt, model=model_name, temperature=args.temperature, tools=None)
        with Pool(initializer=init_worker, initargs=(counter, start_time)) as pool:
            responses = list(tqdm.tqdm(pool.imap(pool_fn, inputs), total=len(inputs), desc="Evaluating (Parallel)"))
    else:
        print("Processing evaluations sequentially...")
        for final_messages in tqdm.tqdm(inputs, desc="Evaluating (Sequential)"):
            generated_text = request_gpt(
                messages=final_messages,
                model=model_name,
                temperature=args.temperature,
                tools=None
            )
            responses.append(generated_text)
    
    # Process responses (which are ChatCompletionMessage objects)
    for job, response_message in zip(evaluate_jobs, responses):
        generated_text = response_message.content or ""
        job["verdict_response"] = generated_text
        job["predicted_ranking"] = extract_final_ranking(generated_text)

# --- Main Function ---
def main():
    """Main execution function."""
    args = parse_args()
    MAX_TURNS = 5

    # --- Initialization ---
    print(f"Arguments: {args}")

    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f if line.strip()] #[:10] # Uncomment for testing

    # --- Prompt Templates ---
    # Updated prompt to be simpler, relies on native tool calling
    ranking_search_prompt = f"""Your task is to determine the correct factual ranking of the provided answers.
Question: {{question}}
{{answers_block}}

Based on the question and answers, think and identify what information you need.
You have tools available to search Wikipedia (search_local) and Google (search_web). You can use them as many times as you want.
When you have enough information, respond with the *exact* string "READY_FOR_EVALUATION".
"""
    # --- Job Initialization ---
    jobs = []
    for i, item in enumerate(input_data):
        job_base = {"id": i, "original_item": item, "search_count": 0, "search_results": []}
        if args.mode == "direct_gen":
            jobs.append({**job_base, "current_step": "evaluate"})
            continue
        
        # Default to ranking scheme
        answers_block = "\n".join([f"Answer{i+1}: {ans['answer']}" for i, ans in enumerate(item.get("answers", []))])
        content = ranking_search_prompt.format(question=item["question"], answers_block=answers_block)
        
        jobs.append({**job_base, "messages": [{"role": "user", "content": content}], "current_step": "decision"})

    # --- Agentic Loop (Decide -> Execute) ---
    if args.mode == "retrieval":
        print("Initializing search APIs: Local (Wiki) and Searxng (Google)...")
        local_api = SearchAPILocal()
        searxng_api = SearchAPISearxng(use_cache=(not args.disable_cache_for_serper))

        for turn in tqdm.tqdm(range(MAX_TURNS), desc="Agent Turns"):
            # Stage 1: DECIDE - Decide next action for jobs in 'decision' state
            jobs_to_decide = [j for j in jobs if j.get("current_step") == "decision"]
            if not jobs_to_decide:
                print(f"No jobs in 'decision' state at turn {turn+1}.")
            else:
                process_decision_stage(jobs_to_decide, args.model_name, args)

            # Stage 2: EXECUTE - Run tool calls generated in the decision stage
            jobs_to_execute = [j for j in jobs if j.get("current_step") == "tool_execution"]
            if not jobs_to_execute:
                print(f"No jobs in 'tool_execution' state at turn {turn+1}.")
            else:
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
    print("--- Starting Final Evaluation Stage for Ranking ---")
    evaluation_jobs = [j for j in jobs if j.get("current_step") == "evaluate"]
    process_evaluation_stage_ranking(evaluation_jobs, args.model_name, args)
    print("--- Final Evaluation Finished ---")

    print("All sequences processed. Saving results...")
    final_results = []
    output_path = args.output_file

    # Prepare the final results list for ranking
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

    # Perform final evaluation
    evaluate_final_results_ranking(final_results)

    # Print final summary
    print(f"\nSummary:\nDataset: {os.path.basename(args.input_file)}\n"
          f"Model: {args.model_name}\n"
          f"Mode: {args.mode}\nScheme: ranking")


if __name__ == "__main__":
    main()