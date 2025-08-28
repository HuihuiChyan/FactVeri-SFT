import argparse
import json
import logging
import re
import requests
from typing import List, Dict
import os
import asyncio
import tqdm

# 导入 sglang 相关的库
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


# --- 命令行参数解析 ---
def parse_args():
    """解析命令行参数，这部分保持不变。"""
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
        default="pairwise",
        choices=("pairwise", "pointwise")
    )
    return parser.parse_args()


# --- 工具定义与实现 ---
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
    工具的具体实现。接受一个查询列表以支持批量操作。
    """
    if not queries:
        return []
    payload = {"queries": queries, "topk": 3, "return_scores": True}
    try:
        response = requests.post(
            "http://127.0.0.1:8000/retrieve", json=payload, timeout=20
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


# --- 结果评估函数 ---
def extract_final_verdict(model_generated_output: str, scheme) -> str:
    """
    从模型生成的最终输出中提取结论。
    """
    answer_pattern = re.compile(r"<verdict>(.*?)</verdict>", re.DOTALL)
    matches = answer_pattern.findall(model_generated_output)
    if matches:
        last_answer = matches[-1].strip().lower()
        if scheme == "pointwise":
            if "not real" in last_answer:
                return 0
            if "real" in last_answer:
                return 1
        elif scheme == "pairwise":
            if last_answer == "answer1":
                return 0
            if last_answer == "answer2":
                return 1
    return -1


def extract_evaluation_result(model_generated_output: str) -> Dict[str, str]:
    """
    从模型生成的评估输出中提取评估结果。
    """
    evaluation_pattern = re.compile(r"<evaluation>(.*?)</evaluation>", re.DOTALL)
    matches = evaluation_pattern.findall(model_generated_output)
    
    if matches:
        evaluation_text = matches[-1].strip()
        useful_match = re.search(r"Useful:\s*(Yes|No)", evaluation_text, re.IGNORECASE)
        useful = useful_match.group(1).lower() == "yes" if useful_match else False
        summary_match = re.search(r"Summary:\s*(.*)", evaluation_text, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else "No summary provided."
        return {"useful": useful, "summary": summary}
    
    return {"useful": False, "summary": "No evaluation found."}


def check_ready_for_evaluation(model_generated_output: str) -> bool:
    """
    检查模型是否表示准备好进行评估（不需要更多搜索）。
    """
    return "READY_FOR_EVALUATION" in model_generated_output


def evaluate_final_results(results: List[Dict], scheme):
    """
    计算并打印评估指标。
    """
    y_true, y_pred = [], []
    invalid_predictions = 0

    for item in results:
        true_label = item.get("verify_result", "")
        pred_label = item.get("final_verdict", "")

        if pred_label not in [0, 1]:
            pred_label = 1 - true_label
            invalid_predictions += 1

        y_true.append(true_label)
        y_pred.append(pred_label)

    if not y_true:
        logging.error("Evaluation failed. No valid ground truth labels found.")
        return None

    metrics_dict = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average='binary', zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average='binary', zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, average='binary', zero_division=0), 4),
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

    batched_input_ids = [
        input_ids[i:i + BATCH_SIZE] 
        for i in range(0, len(input_ids), BATCH_SIZE)
    ]
    results = []
    for input_ids_batch in tqdm.tqdm(batched_input_ids, desc="Batched Generating"):
        # 每次只处理一个批次的输入ID
        results_batch = engine.generate(
            input_ids=input_ids_batch,
            sampling_params=sampling_params
        )
        
        # 将当前批次的结果添加到总结果列表中
        results.extend(results_batch)
    
    return results

def process_search_stage(search_jobs, engine, tokenizer, parser, turn):
    """处理需要生成搜索查询的作业 (iterative, one query per turn)."""
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
    # search_responses = engine.generate(input_ids=search_input_ids, sampling_params=sampling_params)
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
        # API call for a single query
        # Append the tool result to the message history for the next turn
        search_jobs[id]["messages"].append({"role": "tool", "content": api_result})
        # Keep the job in the search stage for the next iteration
        search_jobs[id]["current_step"] = "search"
        search_jobs[id]["search_results"].append({"query": search_jobs[id]["search_query"], "result": api_result})
        del search_jobs[id]["search_query"]

def process_evaluation_stage(evaluate_jobs, engine, tokenizer, args):
    """Summarize search results for jobs that performed searches."""

    def parse_info_tags(text: str) -> List[str]:
        """使用正则表达式从文本中提取所有被 <Info>...</Info> 标签包裹的内容。"""
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

    summarization_prompt = """Based on the preceding conversation, please provide a concise summary of the key facts you have gathered to help verify the factuality of the answer.
Please direct output the key facts with the format of '<Info> fact 1 </Info> <Info> fact 2 </Info> <Info> fact 3 </Info>', without any openings, closings or additional explanations.
If there is no useful information, you can direct output '<Info> No useful information retrieved. </Info>'""" 
    
    summary_input_ids = []
    for job in jobs_to_summarize:
        summary_messages = job["messages"] + [{"role": "user", "content": summarization_prompt}]
        summary_input_ids.append(tokenizer.apply_chat_template(
            conversation=summary_messages, tokenize=True, add_generation_prompt=True
        ))

    sampling_params = {"max_new_tokens": 1024, "temperature": 0}
    # summary_responses = engine.generate(input_ids=summary_input_ids, sampling_params=sampling_params)
    summary_responses = batched_sglang_generation(
                            input_ids=summary_input_ids,
                            sampling_params=sampling_params,
                            engine=engine,
                        )

    for job, response in zip(jobs_to_summarize, summary_responses):
        generated_summary = response["text"]

        extracted_facts = parse_info_tags(generated_summary)
        job["extracted_facts"] = extracted_facts  # 将提取出的事实列表存入 job 字典

        job["current_step"] = "verdict"

def process_verdict_stage(verdict_jobs, engine, tokenizer, args):
    """处理准备好进行最终裁决的作业。"""
    if not verdict_jobs:
        return
    
    if args.scheme == "pairwise":
        verdict_prompt = """You are an expert fact-checking assistant. Your task is to determine which answer is more factually correct.

Question: {question}
Answer1: {answer1}
Answer2: {answer2}
Retrieved Reference Information: {search_summary}

Based on the question, both answers, and the reference information retrieved before, determine which answer is more factually correct.
Please first explantion, and then provide your final verdict with the format: 'Therefore, the answer with better factuality correctness is: <verdict> Answer1/Answer2 </verdict>'. For example, 'Therefore, the answer with better factuality correctness is: <verdict> Answer1 </verdict>"""
    else:
        verdict_prompt = """You are an expert fact-checking assistant. Your task is to determine whether the answer is factually correct or not.

Question: {question}
Answer: {answer}
Retrieved Reference Information: {search_summary}

Based on the question, answer, and the reference information retrieved before, determine if the answer is factually correct or not.
Please first explantion, and then provide your final verdict with the format: 'Therefore, the final verdict: <verdict> Real/Not Real </verdict>'. For example, 'Therefore, the final verdict: <verdict> Real </verdict>"""

    print(f"--- Processing {len(verdict_jobs)} verdict jobs ---")
    verdict_input_ids = []
    for job in verdict_jobs:

        facts = job.get("extracted_facts", [])
        if facts == []:
            formatted_facts = "No useful information retrieved."
        else:
            formatted_facts = " ".join([f"{i}. {fact}" for i,fact in enumerate(facts)])

        if args.scheme == "pointwise":
            prompt = verdict_prompt.format(
                question=job["original_item"]["question"],
                answer=job["original_item"]["answer"]["answer"],
                search_summary=formatted_facts
            )
        else:
            prompt = verdict_prompt.format(
                question=job["original_item"]["question"],
                answer1=job["original_item"]["answer1"]["answer"],
                answer2=job["original_item"]["answer2"]["answer"],
                search_summary=formatted_facts
            )
        # Start a clean message history for the final verdict
        verdict_messages = [{"role": "user", "content": prompt}]
        verdict_input_ids.append(tokenizer.apply_chat_template(
            conversation=verdict_messages, tokenize=True, add_generation_prompt=True
        ))

    sampling_params = {"max_new_tokens": 1024, "temperature": 0}
    # verdict_responses = engine.generate(input_ids=verdict_input_ids, sampling_params=sampling_params)
    verdict_responses = batched_sglang_generation(
                            input_ids=verdict_input_ids,
                            sampling_params=sampling_params,
                            engine=engine,
                        )
    
    for job, response in zip(verdict_jobs, verdict_responses):
        job["verdict_response"] = response["text"]

# --- 主函数 ---
def main():
    """主执行函数，使用SGLang重构。"""
    args = parse_args()
    MAX_TURNS = 5 # A more reasonable max for iterative search

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

    # --- Prompt模板 ---
    if args.scheme == "pointwise":
        search_prompt = f"""You are an expert fact-checking assistant. Your goal is to determine whether the following answer is real or not.

Based on the question and answer below, and any existing search results, identify what additional knowledge you need to verify the factual accuracy. Generate search queries for the missing information.

Question: {{question}}
Answer: {{answer}}

Please analyze what information is still needed and generate search queries one by one using <tool_call> query </tool_call>. The search results will be returned between <tool_response> and </tool_response>.

If you believe you have enough information to make a judgment and don't need search any more, respond with "READY_FOR_EVALUATION" instead of making tool calls."""
    elif args.scheme == "pairwise":
        search_prompt = f"""You are an expert fact-checking assistant. Your task is to determine which of the two provided answers is more factually correct.

Based on the question and answers below, and any existing search results, identify what additional knowledge you need to compare their factual accuracy, and generate search queries for the missing information.

Question: {{question}}
Answer1: {{answer1}}
Answer2: {{answer2}}

Please analyze what information is still needed and generate search queries one by one using <tool_call> query </tool_call>. The search results will be returned between <tool_response> and </tool_response>.

If you believe you have enough information to make a judgment and don't need search any more, respond with "READY_FOR_EVALUATION" instead of making tool calls."""

    # --- 初始化工作队列 ---
    jobs = []
    for i, item in enumerate(input_data):
        job_base = {"id": i, "original_item": item, "search_count": 0}
        if args.mode == "local_retrieval":
            if args.scheme == "pointwise":
                content = search_prompt.format(
                    question=item["question"], 
                    answer=item["answer"]['answer'],
                )
            else:
                content = search_prompt.format(
                    question=item["question"], 
                    answer1=item["answer1"]['answer'],
                    answer2=item["answer2"]['answer'],
                )
            jobs.append({**job_base, "messages": [{"role": "user", "content": content}], "current_step": "search", "search_results": []})

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
    process_evaluation_stage(jobs, engine, tokenizer, args)
    print("--- Evaluation Stage Finished ---")
    
    # Stage 3: Verdict
    print("--- Starting Verdict Stage ---")
    for job in jobs:
        assert job["current_step"] == "verdict"
    process_verdict_stage(jobs, engine, tokenizer, args)
    print("--- Verdict Stage Finished ---")
    
    # --- 结果处理与保存 ---
    print("All sequences processed. Saving results...")
    final_results = []
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for job in sorted(jobs, key=lambda x: x["id"]):
            final_verdict = extract_final_verdict(job["verdict_response"], scheme=args.scheme)
            result_item = {
                **job["original_item"],
                "search_messages": job["messages"],
                "extracted_facts": job["extracted_facts"],
                "verdict_response": job["verdict_response"],
                "final_verdict": final_verdict,
                "search_count": job["search_count"],
            }
            final_results.append(result_item)
            f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")

    print(f"Results saved to {args.output_file}")
    
    input_file_name = os.path.basename(args.input_file)
    model_name = os.path.basename(args.model_path)
    print(f"\nDataset: {input_file_name}\nModel: {model_name}\nMode: {args.mode}")
    
    print("Starting evaluation...")
    metrics = evaluate_final_results(final_results, scheme=args.scheme)
    if metrics:
        print(f"Evaluation metrics: {json.dumps(metrics, indent=4)}")


if __name__ == "__main__":
    main()