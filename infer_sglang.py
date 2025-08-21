import argparse
import json
import logging
import re
import requests
from typing import List, Dict, Any
import os

# 导入 sglang 相关的库
import sglang as sgl
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import Tool, Function

from tqdm import tqdm
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
        # required=True,
        default="/home/huanghui/models/Qwen_Qwen2.5-7B-Instruct",
        help="Path to the SGLang-compatible model.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        # required=True,
        default="/home/huanghui/Search-R1/fact_checking_dataset/bamboogle_judged.jsonl",
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        # required=True,
        default="/home/huanghui/Search-R1/results/bamboogle_judged-Qwen_Qwen2.5-7B-Instruct-local_retrieval.jsonl",
        help="Path to save the output JSONL file.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        # required=True,
        default="/home/huanghui/Search-R1/log/run_bamboogle_judged-Qwen_Qwen2.5-7B-Instruct-local_retrieval.log",
        help="Path to the log file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        # required=True,
        choices=["local_retrieval", "direct_gen"],
        default="local_retrieval",
        help="Set the operating mode: 'local_retrieval' for search-based generation, 'direct_gen' for direct answering.",
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


def search_api_call(query: str) -> str:
    """
    这是工具的具体实现。它调用外部API执行搜索。
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
        return "Search failed due to connection error."

    def _passages2string(retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item.get("document", {}).get("contents", "")
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


# 将工具实现映射到一个字典，方便后续调用。
AVAILABLE_TOOLS = {
    "search": search_api_call,
}


# --- 结果评估函数 ---
def extract_final_verdict(model_generated_output: str) -> str:
    """
    从模型生成的最终输出中提取结论，处理的是最后一条助手的消息。
    """
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = answer_pattern.findall(model_generated_output)
    if matches:
        last_answer = matches[-1].strip().lower()
        if "not real" in last_answer:
            return "Unsupported"
        if "real" in last_answer:
            return "Supported"
    return "Inconclusive"


def evaluate_final_results(results: List[Dict]):
    """
    计算并打印评估指标。
    """
    y_true, y_pred = [], []
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
                y_pred.append(1 - label_map[true_label_str])
        else:
            logging.warning(
                f"Skipping evaluation for an item due to invalid ground truth label: {item.get('label')}"
            )

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
    MAX_SEARCH_ATTEMPTS = 100  # 设置一个最大搜索次数，防止无限循环。（这里不应该设置，对于不同问题需要的长度不一样，进死循环的问题会被最大模型长度限制）

    # --- 日志配置 ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(args.log_file, mode="w")],
    )
    logging.info(f"Arguments: {args}")
    print(f"Logs are being written to {args.log_file}.")

    # --- SGLang引擎和分词器初始化 ---
    logging.info("Initializing SGLang engine...")

    # 最佳实践：通过环境变量 CUDA_VISIBLE_DEVICES=0,1,2 来指定使用哪些GPU。
    # 下面的 tensor_parallel_size 应与可见GPU的数量匹配。
    # 例如，如果你设置了 CUDA_VISIBLE_DEVICES=0,1,2，那么 tensor_parallel_size 应该是 3。
    # 如果未设置环境变量，将尝试使用所有可用的GPU。
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
        # mem_fraction_static=0.8,  # 显存开销也改名了，叫这个，但是好像和vllm的显存计算方式不太一样，这东西控制的不是显存上限，而是和模型本身大小有关的某种东西，都不设置这个选项的话，28G显存够跑7B但是不够跑3B，比较奇怪
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

    # --- 数据加载---
    logging.info(f"Loading data from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f if line.strip()]

    # --- 构建Prompt模板 ---
    # Prompt需要调整，以适应通用工具调用的风格，而不是描述特定的XML标签。在这对<tool_call>和<tool_response>另做描述不知道会咋样。
    if args.mode == "local_retrieval":
        prompt_template = f"""You are an expert fact-checking assistant. Your goal is to determine whether the following response is real or not. You must conduct reasoning first every time you get new information. 
After reasoning, if you find you lack some knowledge, you can call a search engine by <tool_call> query </tool_call> and it will return the top searched results between <tool_response> and </tool_response>. You can search as many times as your want.
If you find no further external knowledge needed, you can directly provide the answer 'Real' or 'Not Real' inside <answer> and </answer>, without detailed illustrations. For example, <answer> Real </answer>. 
Now, begin your work for the following response: {{response}}"""
    else:  # direct_gen 模式
        prompt_template = f"""You are an expert fact-checking assistant. Your goal is to determine whether the following response is real or not. You must conduct reasoning first.
After reasoning, you must provide the answer 'Real' or 'Not Real' inside <answer> and </answer>, without detailed illustrations. For example, <answer> Real </answer>.
Now, begin your work for the following response: {{response}}"""

    # --- 推理循环 ---
    # 这里的逻辑被大大简化了。我们不再需要手动管理active_sequences。
    # 我们为每个输入项独立运行一个完整的对话流程。
    completed_sequences = []
    logging.info(
        f"Starting batch inference for {len(input_data)} items in '{args.mode}' mode..."
    )

    for i, item in enumerate(tqdm(input_data, desc="Processing items")):
        input_content = prompt_template.format(response=item["response"])
        messages = [
            {
                "role": "user",
                "content": input_content,
            },
        ]
        search_count = 0

        # 对于每个条目，我们可能会进行多次模型调用（如果需要工具）
        for turn in range(MAX_SEARCH_ATTEMPTS):
            # 1. 将对话历史转换为模型输入
            # SGLang 使用 apply_chat_template 来处理对话格式，这很标准。
            if args.mode == "local_retrieval":
                input_ids = tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    tools=SEARCH_TOOL_DEFINITION,
                )
            else:
                input_ids = tokenizer.apply_chat_template(
                    conversation=messages, tokenize=True, add_generation_prompt=True
                )

            # 2. 定义采样参数
            sampling_params = {
                "max_new_tokens": 1024,
                "temperature": 0,
                "top_p": 0.95,
                "skip_special_tokens": False,
                # SGLang的工具调用不需要手动设置停止词, 框架会自动处理
            }

            # 3. 使用SGLang引擎生成
            # 我们一次只处理一个序列，但SGLang内部会进行批处理优化。
            # 这是离线推理的核心。
            result = engine.generate(
                input_ids=[input_ids], sampling_params=sampling_params
            )[0]
            # 带个切片没毛病，result的长度是2，结构是{'text': '', 'meta_info': {}}

            generated_text = result["text"]

            logging.info(
                f"--- Item {i} Turn {turn+1} Raw Output ---\n{generated_text}\n---"
            )

            # 4. 解析SGLang的输出
            # `parse_non_stream` 会分离思想、工具调用和正常文本。
            normal_text, calls = parser.parse_non_stream(generated_text)

            # 将模型的思考和文本输出添加到历史记录中
            # assistant_message = {"role": "assistant", "content": normal_text}
            assistant_message = {"role": "assistant", "content": generated_text}
            messages.append(assistant_message)  # 必须在这里直接拼上模型全部输出

            if not calls:
                # 如果没有工具调用，说明模型决定直接回答，循环结束。
                # messages.append(assistant_message)
                logging.info(f"Item {i}: No tool call detected. Finishing.")
                break

            # 5. 如果有工具调用，执行它们
            logging.info(f"Item {i}: Tool calls detected: {calls}")

            # 将工具调用信息也加入到助手的消息中 -> 这是多此一举了
            # assistant_message["tool_calls"] = [call.model_dump() for call in calls]
            # messages.append(assistant_message) # 错误的，一条 message 就允许两个键值对，添加第三个就报错

            for tool_call in calls:
                tool_name = tool_call.name
                tool_args = json.loads(tool_call.parameters)

                if tool_name in AVAILABLE_TOOLS:
                    search_count += 1
                    function_to_call = AVAILABLE_TOOLS[tool_name]
                    tool_result = function_to_call(**tool_args)

                    # 将工具执行结果添加到对话历史中，准备下一次生成
                    messages.append(
                        {
                            "role": "tool",
                            "content": tool_result,
                            # "tool_call_id": tool_call.id, # 离线推理里面没有这个参数
                            "name": tool_name,
                        }
                    )
                    logging.info(
                        f"Item {i}: Executed tool '{tool_name}' with result."
                    )
                else:
                    logging.warning(
                        f"Item {i}: Model called an unknown tool: '{tool_name}'"
                    )

        # 将完成的序列及其所有信息保存下来
        completed_sequences.append(
            {
                "id": i,
                "original_item": item,
                "full_trace_messages": messages,
                "search_count": search_count,
            }
        )

    # --- 结果处理与保存 ---
    logging.info("All sequences processed. Saving results...")
    final_results = []
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for seq in sorted(completed_sequences, key=lambda x: x["id"]):
            if args.mode == "local_retrieval":
                # 将消息列表转换为纯文本轨迹以保持输出格式兼容性
                full_trace_text = tokenizer.apply_chat_template(
                    conversation=seq["full_trace_messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                full_trace_text = tokenizer.apply_chat_template(
                    conversation=seq["full_trace_messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                    tools=SEARCH_TOOL_DEFINITION,
                )
            
            # 提取最终结论时，只看最后一条助手消息
            final_assistant_message = ""
            for msg in reversed(seq["full_trace_messages"]):
                if msg["role"] == "assistant" and msg.get("content"):
                    final_assistant_message = msg["content"]
                    break

            final_verdict = extract_final_verdict(final_assistant_message)

            result_item = {
                **seq["original_item"],
                "model_output_trace": full_trace_text,
                "final_verdict": final_verdict,
                "search_count": seq["search_count"],
            }
            final_results.append(result_item)
            f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")

    logging.info(f"Results saved to {args.output_file}")

    # --- 评估 ---
    logging.info("Starting evaluation...")
    metrics = evaluate_final_results(final_results)
    if metrics:
        logging.info(f"Evaluation metrics: {json.dumps(metrics, indent=4)}")


if __name__ == "__main__":
    main()
