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
        # default="/home/huanghui/models/Qwen_Qwen2.5-7B-Instruct",
        help="Path to the SGLang-compatible model.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        # default="/home/huanghui/Search-R1/fact_checking_dataset/bamboogle_test.jsonl",
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        # default="/home/huanghui/Search-R1/results/bamboogle_test-Qwen_Qwen2.5-7B-Instruct-local_retrieval.jsonl",
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
        # default="local_retrieval",
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
# 遵循Qwen/OpenAI的工具定义风格，将'search'函数定义为一个工具。
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
    这里的API调用仍然是串行的，后端是否支持并发未知。
    """
    if not queries:
        return []
    payload = {"queries": queries, "topk": 3, "return_scores": True}
    try:
        response = requests.post(
            "http://127.0.0.1:8000/retrieve", json=payload, timeout=20
        )
        response.raise_for_status()
        # API返回一个结果列表的列表，每个内部列表对应一个查询
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

    # 为每个查询的结果格式化为字符串
    return [_passages2string(results) for results in results_list]


# 将工具实现映射到一个字典，方便后续调用。
AVAILABLE_TOOLS = {
    "search": search_api_call,
}


# --- 结果评估函数 ---
def extract_final_verdict(model_generated_output: str, scheme) -> str:
    """
    从模型生成的最终输出中提取结论，处理的是最后一条助手的消息。
    """
    answer_pattern = re.compile(r"<verdict>(.*?)</verdict>", re.DOTALL)
    matches = answer_pattern.findall(model_generated_output)
    if matches:
        if scheme == "pointwise":
            last_answer = matches[-1].strip().lower()
            if "not real" in last_answer:
                return "Unsupported"
            if "real" in last_answer:
                return "Supported"
        elif scheme == "pairwise":
            last_answer = matches[-1].strip().lower()
            if last_answer == "answer1":
                return 0
            if last_answer == "answer2":
                return 1
    return -1


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
        y_true.append(true_label)
        y_pred.append(pred_label)

    if not y_true:
        logging.error("Evaluation failed. No valid ground truth labels found.")
        return None

    metrics_dict = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "evaluated_count": len(y_true),
        "invalid_predictions": invalid_predictions,
        "invalid_ratio": round(invalid_predictions / len(y_true) if y_true else 0, 4),
    }

    print("\n--- Evaluation Results ---")
    for key, value in metrics_dict.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("--------------------------\n")
    return metrics_dict


# --- 主函数 ---
def main():
    """主执行函数，使用SGLang重构。"""
    args = parse_args()
    MAX_TURNS = 20  # 设置一个最大轮次，防止无限循环。

    # --- 日志配置 ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"Arguments: {args}")

    # --- SGLang引擎和分词器初始化 ---
    logging.info("Initializing SGLang engine...")

    # 尝试通过环境变量 CUDA_VISIBLE_DEVICES=0,1,2 来指定使用哪些GPU。
    # 下面的 tensor_parallel_size 应与可见GPU的数量匹配。
    # 例如，如果你设置了 CUDA_VISIBLE_DEVICES=0,1,2，那么 tensor_parallel_size 应该是 3。
    # 如果未设置环境变量，将尝试使用所有可用的GPU。
    # 这部分代码应该不会影响默认设置（无设置）下的程序行为
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        tp_size = len(cuda_visible_devices.split(","))
    else:
        # 如果不设置，SGLang默认会使用所有可见的GPU
        import torch

        tp_size = torch.cuda.device_count()

    engine = sgl.Engine(
        model_path=args.model_path,
        trust_remote_code=True,
        # tp_size=tp_size, # 张量并行改名了，叫这个
        mem_fraction_static=0.8,  # 显存开销也改名了，叫这个，但是好像和vllm的显存计算方式不太一样，这东西控制的不是显存上限，而是和模型本身大小有关的某种东西，都不设置这个选项的话，28G显存够跑7B但是不够跑3B，比较奇怪
    )
    tokenizer = engine.tokenizer_manager.tokenizer

    # SGLang工具调用解析器
    # 这是处理工具调用的核心
    # tools = [
    #     Tool(type=t["type"], function=Function(**t["function"]))
    #     for t in [SEARCH_TOOL_DEFINITION]
    # ] # 这个实现比下面的函数更简单但是有点问题，以后再改
    def convert_dict_to_tool(tool_dict: dict) -> Tool:
        function_dict = tool_dict.get("function", {})
        return Tool(
            type=tool_dict.get("type", "function"),
            function=Function(
                name=function_dict.get("name"),
                description=function_dict.get("description"),
                parameters=function_dict.get("parameters"),
            ),
        )

    tools = [convert_dict_to_tool(raw_tool) for raw_tool in SEARCH_TOOL_DEFINITION]
    parser = FunctionCallParser(tools=tools, tool_call_parser="qwen25")

    # --- 数据加载 ---
    logging.info(f"Loading data from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f if line.strip()]

    # --- Prompt模板 ---
    if args.mode == "local_retrieval":
        if args.scheme == "pointwise":
            prompt_template = f"""You are an expert fact-checking assistant. Your goal is to determine whether the following answer is real or not. You must conduct reasoning first every time you get new information. 
After reasoning, if you find you lack some knowledge, you can call a search engine by <tool_call> query </tool_call> and it will return the top searched results between <tool_response> and </tool_response>. You can search as many times as your want.
If you find no further external knowledge needed, you can directly provide the final verdict with the format of 'Therefore, the final verdict is: <verdict> Real/Not Real </verdict>'. For example, 'Therefore, the final verdict is: <verdict> Real </verdict>'. 
Now, begin your work for the following question and answers:
Question: {{question}}
Answer: {{answer}}"""
        elif args.scheme == "pairwise":
            prompt_template = f"""You are an expert fact-checking assistant. Your task is to determine which of the two provided answers is more factually correct. You must conduct reasoning first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <tool_call> query </tool_call> and it will return the top searched results between <tool_response> and </tool_response>. You can search as many times as your want.
f you find no further external knowledge needed, you can directly provide the final verdict with the format of 'Therefore, the answer with better factuality correctness is: <verdict> Answer1/Answer2 </verdict>'. For example, 'Therefore, the answer with better factuality correctness is: <verdict> Answer1 </verdict>'.
Now, begin your work for the following question and answers:
Question: {{question}}
Answer1: {{answer1}}
Answer2: {{answer2}}"""
    else:
        if args.scheme == "pointwise":
            prompt_template = f"""You are an expert fact-checking assistant. Your goal is to determine whether the following answer is real or not. You must conduct reasoning first.
After reasoning, you must directly provide the final verdict with the format of 'Therefore, the final verdict is: <verdict> Real/Not Real </verdict>'. For example, 'Therefore, the final verdict is: <verdict> Real </verdict>'.
Now, begin your work for the following question and answer:
Question: {{question}}
Answer: {{answer}}"""
        elif args.scheme == "pairwise":
            prompt_template = f"""You are an expert fact-checking assistant. Your task is to determine which of the two provided answers is more factually correct. You must conduct reasoning first.
After reasoning, you must directly provide the final verdict with the format of 'Therefore, the answer with better factuality correctness is: <verdict> Answer1/Answer2 </verdict>'. For example, 'Therefore, the answer with better factuality correctness is: <verdict> Answer1 </verdict>'.
Now, begin your work for the following question and answers:
Question: {{question}}
Answer1: {{answer1}}
Answer2: {{answer2}}"""
            
    # --- 初始化工作队列 ---
    # 核心的数据结构，用于管理每个任务的状态。
    jobs = []
    for i, item in enumerate(input_data):
        if args.scheme == "pointwise":
            input_content = prompt_template.format(question=item["question"], answer=item["answer"]['answer'])
        elif args.scheme == "pairwise":
            input_content = prompt_template.format(question=item["question"], answer1=item["answer1"]['answer'], answer2=item["answer2"]['answer'])
        jobs.append(
            {
                "id": i,
                "original_item": item,
                "messages": [{"role": "user", "content": input_content}],
                "search_count": 0,
                "status": "active",  # 'active' or 'completed'
            }
        )

    # --- 批处理推理循环 ---
    logging.info(
        f"Starting batched inference for {len(jobs)} items in '{args.mode}' mode..."
    )
    if args.gpu_idx is not None:
        pbar = tqdm.tqdm(total=len(jobs), desc=f"Completed Items at GPU {args.gpu_idx}")
    else:
        pbar = tqdm.tqdm(total=len(jobs), desc=f"Completed Items")
    for turn in range(MAX_TURNS):
        active_jobs = [job for job in jobs if job["status"] == "active"]
        if not active_jobs:
            logging.info("All jobs completed. Exiting main loop.")
            break

        logging.info(
            f"--- Turn {turn + 1}: Processing {len(active_jobs)} active jobs ---"
        )

        # 1. 准备生成批次
        input_ids_all = []
        for job in active_jobs:
            if args.mode == "local_retrieval":
                input_ids = tokenizer.apply_chat_template(
                    conversation=job["messages"],
                    tokenize=True,
                    add_generation_prompt=True,
                    tools=SEARCH_TOOL_DEFINITION,
                )
            else:
                input_ids = tokenizer.apply_chat_template(
                    conversation=job["messages"],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            input_ids_all.append(input_ids)

        # 2. 批量生成
        sampling_params = {"max_new_tokens": 1024, "temperature": 0}

        BATCH_SIZE = 100
        batched_input_ids = [
            input_ids_all[i:i + BATCH_SIZE] 
            for i in range(0, len(input_ids_all), BATCH_SIZE)
        ]

        # 使用 tqdm 包装批次列表，以显示进度条
        results = []
        if args.gpu_idx is not None:
            tqdm_desc = f"Batched Generating at GPU {args.gpu_idx}"
        else:
            tqdm_desc = "Batched Generating"
        for input_ids_batch in tqdm.tqdm(batched_input_ids, desc=tqdm_desc):
            # 每次只处理一个批次的输入ID
            results_batch = engine.generate(
                input_ids=input_ids_batch,
                sampling_params=sampling_params
            )
            
            # 将当前批次的结果添加到总结果列表中
            results.extend(results_batch)

        # 3. 处理结果并准备工具调用
        jobs_requiring_tool_calls = []
        for job, result in zip(active_jobs, results):
            generated_text = result["text"]
            job["messages"].append({"role": "assistant", "content": generated_text})

            # 解析工具调用,并捕获可能的JSON解析错误
            try:
                _, calls = parser.parse_non_stream(generated_text)
            except json.JSONDecodeError:
                logging.warning(
                    f"Job {job['id']}: Failed to parse tool call from model output. Treating as no-call."
                )
                logging.warning(
                    f"Problematic output for Job {job['id']}:\n---\n{generated_text}\n---"
                )
                calls = []  # 将其视作没有工具调用

            if not calls or args.mode == "direct_gen":
                # 没有工具调用，或者在 direct_gen 模式下，任务完成
                job["status"] = "completed"
                pbar.update(1)
            else:
                # 存在工具调用，将其加入待处理列表
                job["tool_calls"] = calls
                jobs_requiring_tool_calls.append(job)

        print(f"Start searching on {len(jobs_requiring_tool_calls)} items!")
        # 4. 批量执行工具调用
        if jobs_requiring_tool_calls:
            logging.info(
                f"Executing tool calls for {len(jobs_requiring_tool_calls)} jobs."
            )

            # 收集所有查询
            all_queries = []
            for job in jobs_requiring_tool_calls:
                job["search_count"] += len(job["tool_calls"])
                for call in job["tool_calls"]:
                    # 假设只有一个名为 'search' 的工具
                    if call.name == "search":
                        try:
                            query_args = json.loads(call.parameters)
                            all_queries.append(query_args.get("query", ""))
                        except json.JSONDecodeError:
                            logging.warning(
                                f"Job {job['id']}: Malformed parameters in tool call. Skipping call."
                            )

            # 一次性调用搜索API
            # 注意：如果API不支持批量，这里仍然是性能瓶颈。
            # 但逻辑上，我们已经将I/O操作集中处理了。
            tool_results = search_api_call(all_queries)

            # 将结果分发回对应的job
            result_idx = 0
            for job in jobs_requiring_tool_calls:
                for call in job["tool_calls"]:
                    if call.name == "search":
                        job["messages"].append(
                            {
                                "role": "tool",
                                "content": tool_results[result_idx],
                                "name": call.name,
                            }
                        )
                        result_idx += 1
                del job["tool_calls"]  # 清理掉，以免影响下一轮
        print("Finish searching!")

    pbar.close()
    if turn == MAX_TURNS - 1 and any(job["status"] == "active" for job in jobs):
        logging.warning(
            f"Reached max turns ({MAX_TURNS}). Some jobs may not be complete."
        )
        # 将剩余的 active jobs 强制标记为 completed 以便输出
        for job in jobs:
            if job["status"] == "active":
                job["status"] = "completed"

    # --- 结果处理与保存 ---
    logging.info("All sequences processed. Saving results...")
    final_results = []
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for job in sorted(jobs, key=lambda x: x["id"]):
            if args.mode == "local_retrieval":
                # 将消息列表转换为纯文本轨迹以保持输出格式兼容性
                full_trace_text = tokenizer.apply_chat_template(
                    conversation=job["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                    tools=SEARCH_TOOL_DEFINITION,
                )
            else:
                full_trace_text = tokenizer.apply_chat_template(
                    conversation=job["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )

            # 提取最终结论时，只看最后一条助手消息
            final_assistant_message = ""
            for msg in reversed(job["messages"]):
                if msg["role"] == "assistant" and msg.get("content"):
                    final_assistant_message = msg["content"]
                    break

            final_verdict = extract_final_verdict(final_assistant_message, scheme=args.scheme)
            result_item = {
                **job["original_item"],
                "model_output_trace": full_trace_text,
                "final_verdict": final_verdict,
                "search_count": job["search_count"],
            }
            final_results.append(result_item)
            f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")

    logging.info(f"Results saved to {args.output_file}")

    # 提取输入文件的最后一级
    input_file_name = os.path.basename(args.input_file)
    # 提取模型路径的最后一级
    model_name = os.path.basename(args.model_path)

    print(f"\nDataset: {input_file_name}\nModel: {model_name}\nMode: {args.mode}")
    logging.info(f"Dataset: {input_file_name}, Model: {model_name}, Mode: {args.mode}")

    # --- 评估 ---
    logging.info("Starting evaluation...")
    metrics = evaluate_final_results(final_results, scheme=args.scheme)
    if metrics:
        logging.info(f"Evaluation metrics: {json.dumps(metrics, indent=4)}")


if __name__ == "__main__":
    main()
