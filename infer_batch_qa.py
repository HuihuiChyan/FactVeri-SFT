import argparse
import json
import logging
import re
import requests
from typing import List, Dict, Optional
from collections import Counter

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


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
    # 运行模式参数
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
        # 返回最后一个匹配项，去除首尾空白
        return matches[-1].strip()

    # 如果没有找到 <answer> 标签，则返回空字符串表示无效
    return ""


def evaluate_final_results(results: List[Dict]):
    """
    计算并打印评估指标，适用于开放式问答，计算 EM 和 F1。
    """
    total_em = 0.0
    total_f1 = 0.0
    evaluated_count = 0
    invalid_predictions = 0

    def _calculate_f1(prediction: str, ground_truth: str) -> float:
        """计算两个字符串之间的 F1 分数。"""
        prediction_tokens = prediction.split()
        ground_truth_tokens = ground_truth.split()

        if not prediction_tokens or not ground_truth_tokens:
            return 0.0

        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    for item in results:
        # 获取真实答案和预测答案
        true_answer = item.get("answer", "").strip().lower()
        pred_answer = item.get("final_verdict", "").strip().lower()

        # 跳过没有真实答案的样本
        if not true_answer:
            logging.warning(
                f"Skipping item for evaluation due to empty ground truth answer: {item.get('_id')}"
            )
            continue

        evaluated_count += 1

        # 检查预测是否有效 (即是否成功提取答案)
        if not pred_answer:
            invalid_predictions += 1
            continue  # 无效预测的 EM 和 F1 均为 0，所以直接跳过

        # 计算 Exact Match (EM)
        if pred_answer == true_answer:
            total_em += 1

        # 计算 F1 Score
        total_f1 += _calculate_f1(pred_answer, true_answer)

    # 计算最终指标
    final_em = total_em / evaluated_count if evaluated_count > 0 else 0
    final_f1 = total_f1 / evaluated_count if evaluated_count > 0 else 0
    invalid_ratio = invalid_predictions / evaluated_count if evaluated_count > 0 else 0

    metrics_dict = {
        "exact_match": round(final_em, 4),
        "f1_score": round(final_f1, 4),
        "evaluated_count": evaluated_count,
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

    # 这些是模型生成搜索请求或答案时可能出现的停止符
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
        # tensor_parallel_size=2,
        gpu_memory_utilization=0.60,  # 最高使用显存
        trust_remote_code=True,
        max_model_len=32768,
    )

    # --- 数据加载 ---
    logging.info(f"Loading data from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f if line.strip()]

    # --- 构建初始Prompt ---
    # 根据模式选择不同的prompt模板
    if args.mode == "local_retrieval":
        base_prompt_template = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want, but you should only include one single query per search tag. \
After the information is returned, you can continue reasoning or call another search if necessary. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. \
Question: {{question}}\n"""
    else:  # direct_gen 模式
        base_prompt_template = f"""Answer the given question. \
You can conduct reasoning inside <think> and </think> first. \
After thinking, you must provide the answer directly inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. \
Question: {{question}}\n"""
    active_sequences = []
    for i, item in enumerate(input_data):
        question = item.get("question")
        if not question:
            logging.warning(f"Skipping item {i} due to missing 'question' field.")
            continue

        # 格式化初始prompt
        prompt = base_prompt_template.format(question=question)

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
    logging.info(
        f"Starting batch inference for {len(active_sequences)} items in '{args.mode}' mode..."
    )

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

                info_block = f"\n\n<information>{search_results}</information>\n\n"
                current_seq["full_trace"] += info_block
                current_seq["search_count"] += 1
                next_active_sequences.append(current_seq)  # 继续处理这个序列
                logging.info(
                    f"Item {current_seq['id']}: Appended search results and continuing."
                )
            else:
                # 在 direct_gen 模式或 local_retrieval 模式未找到搜索时，任务完成
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
                question=seq["original_item"].get("question")
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

    print(f"\nDataset: {input_file_name}\nModel: {model_name}\nMode: {args.mode}")
    logging.info(f"Dataset: {input_file_name}, Model: {model_name}, Mode: {args.mode}")

    # --- 评估 ---
    logging.info("Starting evaluation...")
    metrics = evaluate_final_results(final_results)
    if metrics:
        logging.info(f"Evaluation metrics: {json.dumps(metrics, indent=4)}")


if __name__ == "__main__":
    main()