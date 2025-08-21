import argparse
import json
import logging
import re
import requests
from typing import List, Dict, Optional

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
)

# --- 常量和特殊标记 ---
# START_THINK = "<think>"
# END_THINK = "</think>"
# START_SEARCH = "<search>"
# END_SEARCH = "</search>"
# START_ANSWER = "<answer>"
# END_ANSWER = "</answer>"
# START_INFO = "<information>"
# END_INFO = "</information>"


# --- 命令行参数解析 ---
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Batch inference script for fact-checking model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the VLLM-compatible model.",
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output JSONL file.",
    )
    parser.add_argument(
        "--log_file", type=str, required=True, help="Path to the log file."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["local_retrieval", "direct_gen"],
        help="Set the operating mode: 'local_retrieval' for search-based generation, 'direct_gen' for direct answering.",
    )
    return parser.parse_args()


# --- 核心功能函数 ---


def get_query(text: str) -> Optional[str]:
    """
    从文本中提取最后一个 <search> 和 </search> 之间的查询。
    """
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    return None


def search(query: str) -> str:
    """
    使用给定的查询调用外部搜索API。
    """
    payload = {"queries": [query], "topk": 3, "return_scores": True}
    try:
        response = requests.post(
            "http://127.0.0.1:8000/retrieve", json=payload, timeout=10
        )
        response.raise_for_status()
        results = response.json()["result"]
    except requests.RequestException as e:
        logging.error(f"Search API request failed for query '{query}': {e}")
        return "Search failed."

    def _passages2string(retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item.get("document", {}).get("contents", "")
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


def extract_final_verdict(model_generated_output: str) -> str:
    """
    从模型生成的输出轨迹中提取最终结论。
    此函数应只处理模型生成的部分，以避免匹配到Prompt中的示例。
    """
    # 优先使用 <answer> 标签提取结论
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = answer_pattern.findall(model_generated_output)
    if matches:
        last_answer = matches[-1].strip().lower()
        if "not real" in last_answer:
            return "Unsupported"
        if "real" in last_answer:
            return "Supported"

    # 如果没有找到明确的 <answer> 标签，则返回不确定
    return "Inconclusive"


def evaluate_final_results(results: List[Dict]):
    """
    计算并打印评估指标，参考了您提供的代码片段。
    """
    y_true = []
    y_pred = []
    invalid_predictions = 0

    label_map = {"supported": 1, "unsupported": 0}

    for item in results:
        true_label_str = item.get("label", "").lower()
        pred_label_str = item.get("final_verdict", "").lower()

        if true_label_str in label_map:
            y_true.append(label_map[true_label_str])
            if pred_label_str in label_map:
                y_pred.append(label_map[pred_label_str])
            else:
                invalid_predictions += 1
                # 将无效预测标记为与真实标签相反的值，以确保其被计为错误
                y_pred.append(1 - label_map[true_label_str])
        else:
            logging.warning(
                f"Skipping item for evaluation due to invalid ground truth label: {item.get('label')}"
            )

    if not y_true:
        logging.error(
            "Evaluation could not be performed. No items with valid ground truth labels found."
        )
        return None

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    invalid_ratio = invalid_predictions / len(y_true) if y_true else 0

    metrics_dict = {
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "evaluated_count": len(y_true),
        "invalid_predictions": invalid_predictions,
        "invalid_ratio": round(invalid_ratio, 4),
    }

    print("\n--- Evaluation Results ---")
    for key, value in metrics_dict.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("--------------------------\n")

    return metrics_dict


def main():
    """主执行函数"""
    args = parse_args()

    # --- 日志配置 ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.log_file, mode="w"),
        ],
    )
    logging.info(f"Arguments: {args}")
    logging.info(f"Log file will be saved to {args.log_file}")
    print(
        f"Logs are being written to {args.log_file}. Evaluation results will be printed here."
    )

    # --- 模型和分词器初始化 ---
    logging.info("Initializing model and tokenizer with VLLM...")

    # Qwen2.5-Instruct系列的EOS tokens
    curr_eos = [151645, 151643]

    # 保留原始 infer.py 的 target_sequences 定义, 用于判停
    # 这些是模型生成搜索请求时可能出现的停止符
    target_sequences = [
        "</search>",
        # " </search>",
        # "</search>\n",
        # " </search>\n",
        # "</search>\n\n",
        # " </search>\n\n",
        "</answer>",
        # 原始代码中没有关于answer的标签，可能因此导致了模型不停止输出。
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        # tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.60,  # 最高使用显存
        trust_remote_code=True,
        max_model_len=32768,
    )

    # --- 数据加载 ---
    logging.info(f"Loading data from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f if line.strip()]

    # --- 构建初始Prompt ---
    # 这个模板保留了原始 infer.py 的核心指令
    if args.mode == "local_retrieval":
        base_prompt_template = f"""You are an expert fact-checking assistant. \
Your goal is to determine whether the following claim is real or not. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want, but you should only include one single query per search tag. \
After the information is returned, you can continue reasoning or call another search if necessary. \
If you find no further external knowledge needed, you can directly provide the answer 'Real' or 'Not Real' inside <answer> and </answer>, without detailed illustrations. \
For example, <answer> Real </answer>. And no more else after that. \
Now, begin your work for the following claim: {{claim}}\n"""
    else:  # direct_gen 模式
        base_prompt_template = f"""You are an expert fact-checking assistant. \
Your goal is to determine whether the following claim is real or not. \
You must conduct reasoning inside <think> and </think> first. \
After reasoning, you can directly provide the answer 'Real' or 'Not Real' inside <answer> and </answer>, without detailed illustrations. \
For example, <answer> Real </answer>. And no more else after that. \
Now, begin your work for the following claim: {{claim}}\n"""
    active_sequences = []
    for i, item in enumerate(input_data):
        claim = item.get("response") or item.get("claim")
        if not claim:
            logging.warning(
                f"Skipping item {i} due to missing 'claim' or 'response' field."
            )
            continue

        # 格式化初始prompt
        prompt = base_prompt_template.format(claim=claim)

        active_sequences.append(
            {
                "id": i,
                "original_item": item,
                "full_trace": prompt,  # 记录完整的交互轨迹
                "finished": False,
                "search_count": 0,
            }
        )

    completed_sequences = []

    # --- 批量推理循环 ---
    logging.info(f"Starting batch inference for {len(active_sequences)} items...")

    while active_sequences:
        # 准备当前批次的prompts
        prompts_to_process = [seq["full_trace"] for seq in active_sequences]

        # 定义VLLM的采样参数

        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=1024,  # 每次生成最多token
            stop=target_sequences
            + [tokenizer.decode(eos) for eos in curr_eos],  # 关键：在这里设置停止条件
            include_stop_str_in_output=True,
        )

        # 使用VLLM进行批量生成
        vllm_outputs = llm.generate(prompts_to_process, sampling_params, use_tqdm=True)

        # 处理生成结果
        next_active_sequences = []
        for i, output in enumerate(vllm_outputs):
            current_seq = active_sequences[i]

            # 将原始输出记录到日志文件
            completion_output = output.outputs[0]
            generated_text = completion_output.text
            finish_reason = completion_output.finish_reason

            logging.info(
                f"--- Item {current_seq['id']} Raw Output ---\n{generated_text}\n--- Stop Reason: {finish_reason} ---"
            )

            current_seq["full_trace"] += generated_text

            # 仅在 local_retrieval 模式下检查是否需要搜索
            query = None
            if args.mode == "local_retrieval":
                query = get_query(generated_text)

            if query:
                logging.info(f"Item {current_seq['id']}: Search query found: '{query}'")
                search_results = search(query)

                # 保留原始 infer.py 的注入模板
                info_block = f"\n\n<information>{search_results}</information>\n\n"
                current_seq["full_trace"] += info_block
                current_seq["search_count"] += 1
                next_active_sequences.append(current_seq)  # 继续处理这个序列
                logging.info(
                    f"Item {current_seq['id']}: Appended search results and continuing."
                )
            else:
                # 没有找到搜索请求，认为此任务已完成
                current_seq["finished"] = True
                completed_sequences.append(current_seq)
                logging.info(
                    f"Item {current_seq['id']}: Finished processing (Reason: {finish_reason})."
                )

        active_sequences = next_active_sequences

        logging.info(
            f"{len(active_sequences)} sequences remain active. {len(completed_sequences)} completed."
        )

    # --- 结果处理与保存 ---
    logging.info("All sequences processed. Saving results...")
    final_results = []
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for seq in sorted(completed_sequences, key=lambda x: x["id"]):
            # 通过设置偏移值切片，只将模型生成的部分传给extract_final_verdict，避免匹配到Prompt中的示例
            initial_prompt = base_prompt_template.format(
                claim=seq["original_item"].get("response")
                or seq["original_item"].get("claim")
            )
            model_generated_output = seq["full_trace"][len(initial_prompt) :]

            final_verdict = extract_final_verdict(model_generated_output)

            result_item = {
                **seq["original_item"],
                "model_output_trace": seq["full_trace"],
                "final_verdict": final_verdict,
                "search_count": seq["search_count"],
            }
            final_results.append(result_item)
            f_out.write(json.dumps(result_item) + "\n")

    logging.info(f"Results saved to {args.output_file}")
    import os

    # 提取输入文件的最后一级
    input_file_name = os.path.basename(args.input_file)
    # 提取模型路径的最后一级
    model_name = os.path.basename(args.model_path)

    print(f"\nDataset: {input_file_name}\nModel: {model_name}")
    logging.info(f"Dataset: {input_file_name}, Model: {model_name}")

    # --- 评估 ---
    logging.info("Starting evaluation...")
    metrics = evaluate_final_results(final_results)
    if metrics:
        logging.info(f"Evaluation metrics: {json.dumps(metrics, indent=4)}")


if __name__ == "__main__":
    main()
