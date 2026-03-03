"""
VeriScore ablation: GPT-4o + Google Search only.
One retrieval per answer; score each answer separately (scoring mode).
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm

# Allow importing from FactVeri-SFT/src when run from project root or ablation/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from search_api_serper import SearchAPISerper
from infer_batch_gpt4 import request_gpt

# Only Google Search tool (no Wikipedia)
SEARCH_WEB_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Performs a web search using Google.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query keyword for the web search engine.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

SCORING_VERDICT_PROMPT_RETRIEVAL = """Based on the preceding conversation, your task is to score the factuality of the given answer from a scale of 1-10.

Question: {question}
Answer: {answer}

Please first provide your explanation, and then state your final verdict in the format: '**Final Verdict**: <verdict> a score between 1-10 </verdict>'. Example: '**Final Verdict**: <verdict> 7 </verdict>'."""


def extract_final_scoring(model_generated_output: str) -> int:
    """Extracts the final score from the model's output."""
    if not isinstance(model_generated_output, str):
        return 0
    match = re.search(r"<verdict>\s*(\d+)\s*</verdict>", model_generated_output, re.IGNORECASE)
    if not match:
        match = re.search(r"\*\*Final Verdict\*\*:\s*<verdict>\s*(\d+)\s*</verdict>", model_generated_output, re.IGNORECASE)
    if not match:
        match = re.search(r"\*\*Final Verdict\*\*:\s*(\d+)\s*", model_generated_output, re.IGNORECASE)
    if not match:
        match = re.search(r"Final Verdict:\s*(\d+)\s*", model_generated_output, re.IGNORECASE)
    try:
        return int(match.group(1))
    except:
        return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="VeriScore: one Google search per answer, then score each answer separately (scoring mode)."
    )
    parser.add_argument("--model_name", type=str, required=True, help="OpenAI model (e.g. gpt-4o).")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL path.")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL path.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--disable_cache_for_serper", action="store_true", default=False)
    parser.add_argument("--max_workers", type=int, default=32, help="Maximum number of threads for parallel processing samples (default: 8)")
    return parser.parse_args()


def run_one_sample_veriscore(job, model_name, temperature, web_api):
    """For one sample: one search per answer, then return job with messages and search_count."""
    item = job["original_item"]
    question = item["question"]
    answers = item.get("answers", [])
    n = len(answers)
    if n == 0:
        job["messages"] = []
        job["search_count"] = 0
        return job

    answers_block = "\n".join([f"Answer{i+1}: {ans['answer']}" for i, ans in enumerate(answers)])
    messages = [
        {
            "role": "user",
            "content": f"""Your task is to verify the factual correctness of each answer using web search.

Question: {question}

{answers_block}

You have only one tool: search_web (Google). For Answer 1 only, generate exactly one search query to verify it. Call the search_web tool once.""",
        }
    ]
    search_count = 0

    for answer_idx in range(n):
        response_message = request_gpt(
            messages=messages,
            model=model_name,
            temperature=temperature,
            tools=SEARCH_WEB_TOOL,
        )
        messages.append(response_message.to_dict())

        query = None
        tool_call_id = None
        if response_message.tool_calls:
            for tc in response_message.tool_calls:
                if tc.function.name == "search_web":
                    try:
                        params = json.loads(tc.function.arguments or "{}")
                        query = params.get("query")
                        tool_call_id = tc.id
                        break
                    except (json.JSONDecodeError, TypeError):
                        pass
            if not tool_call_id and response_message.tool_calls:
                tool_call_id = response_message.tool_calls[0].id

        if query and tool_call_id:
            results = web_api.search_api_call([query])
            content = f"[Source: Google Search]\n{results[0]}" if results else "[Source: Google Search]\nNo results found."
            messages.append({"role": "tool", "content": content, "tool_call_id": tool_call_id})
            search_count += 1
        elif tool_call_id:
            messages.append({
                "role": "tool",
                "content": "[Source: Google Search]\nNo search was performed (no valid query).",
                "tool_call_id": tool_call_id,
            })

        # Next answer: ask for one query for Answer (answer_idx+2)
        if answer_idx + 1 < n:
            messages.append({
                "role": "user",
                "content": f"For Answer {answer_idx + 2} only, generate exactly one search query to verify it. Call the search_web tool once.",
            })

    job["messages"] = messages
    job["search_count"] = search_count
    return job


def score_one_answer_veriscore(args_tuple):
    """Wrapper function for parallel execution of scoring one answer."""
    job_idx, answer_idx, messages, question, answer_text, model_name, temperature = args_tuple
    prompt = SCORING_VERDICT_PROMPT_RETRIEVAL.format(
        question=question,
        answer=answer_text,
    )
    final_messages = messages + [{"role": "user", "content": prompt}]
    response_message = request_gpt(
        messages=final_messages,
        model=model_name,
        temperature=temperature,
        tools=None,
    )
    generated_text = response_message.content or ""
    score = extract_final_scoring(generated_text)
    return job_idx, answer_idx, generated_text, score


def process_evaluation_stage_scoring_veriscore(evaluate_jobs, model_name, temperature, max_workers=None):
    """Append verdict prompt and call GPT; score each answer separately using multithreading."""
    # Prepare all scoring tasks
    scoring_tasks = []
    for job_idx, job in enumerate(evaluate_jobs):
        item = job["original_item"]
        question = item["question"]
        answers = item.get("answers", [])
        messages = job.get("messages", [])
        
        # Initialize predicted_scoring for each answer if not exists
        for ans in answers:
            if "predicted_scoring" not in ans:
                ans["predicted_scoring"] = 0
            if "verdict_response" not in ans:
                ans["verdict_response"] = ""
        
        # Create tasks for each answer
        for answer_idx, answer in enumerate(answers):
            scoring_tasks.append((
                job_idx,
                answer_idx,
                messages,
                question,
                answer["answer"],
                model_name,
                temperature,
            ))
    
    total_answers = len(scoring_tasks)
    if total_answers == 0:
        return
    
    if max_workers is None:
        max_workers = min(total_answers, 8)  # Default to min of answer count and 8 threads
    
    # Process answers in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(score_one_answer_veriscore, task): task for task in scoring_tasks}
        
        # Process completed tasks with progress bar
        for future in tqdm.tqdm(as_completed(future_to_task), total=total_answers, desc="Verdict (VeriScore)"):
            job_idx, answer_idx, generated_text, score = future.result()
            # Update the answer in the original job
            evaluate_jobs[job_idx]["original_item"]["answers"][answer_idx]["verdict_response"] = generated_text
            evaluate_jobs[job_idx]["original_item"]["answers"][answer_idx]["predicted_scoring"] = score


def main():
    args = parse_args()
    print(f"Arguments: {args}")

    with open(args.input_file, "r", encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f if line.strip()]

    jobs = []
    for i, item in enumerate(input_data):
        jobs.append({"id": i, "original_item": item})

    print("--- VeriScore: one search per answer ---")
    
    # Process samples in parallel using ThreadPoolExecutor
    def process_sample(job):
        """Wrapper function for parallel execution of run_one_sample_veriscore."""
        # Each thread needs its own web_api instance for thread safety
        web_api = SearchAPISerper(use_cache=(not args.disable_cache_for_serper))
        return run_one_sample_veriscore(job, args.model_name, args.temperature, web_api)
    
    max_workers = args.max_workers if args.max_workers is not None else 8
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_job = {executor.submit(process_sample, job): job for job in jobs}
        
        # Process completed jobs with progress bar
        for future in tqdm.tqdm(as_completed(future_to_job), total=len(jobs), desc="Samples"):
            future.result()  # Wait for completion and handle any exceptions

    print("--- Final evaluation (scoring verdict) ---")
    process_evaluation_stage_scoring_veriscore(jobs, args.model_name, args.temperature, max_workers=args.max_workers)

    final_results = []
    for job in sorted(jobs, key=lambda x: x["id"]):
        result_item = {
            **job["original_item"],
            "search_messages": job.get("messages", []),
            "search_count": job.get("search_count", 0),
        }
        final_results.append(result_item)

    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for result in final_results:
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Results saved to {args.output_file}")

    print(f"\nSummary: VeriScore (Scoring Mode) | Dataset: {os.path.basename(args.input_file)} | Model: {args.model_name}")


if __name__ == "__main__":
    main()
