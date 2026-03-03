import argparse
import copy
import json
import logging
import re
import requests
import time
from typing import List, Dict
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer

# 导入本地搜索 API 模块
from search_api_local import SearchAPILocal
from search_api_serper import SearchAPISerper


# --- 命令行参数解析 ---
def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Single-threaded scoring inference script using SGLang HTTP API."
    )
    parser.add_argument(
        "--sglang_url", type=str, default=None,
        help="(Optional) URL of the SGLang HTTP server. If set, used for both retrieval and evaluation."
    )
    parser.add_argument(
        "--sglang_url_retrieval", type=str, default="http://localhost:30000",
        help="URL of the SGLang HTTP server for retrieval (search) stage."
    )
    parser.add_argument(
        "--sglang_url_evaluation", type=str, default="http://localhost:30000",
        help="URL of the SGLang HTTP server for evaluation (scoring) stage."
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True,
        help="Path to the tokenizer (for prompt formatting)."
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to save the output JSONL file."
    )
    parser.add_argument(
        "--mode", type=str, required=True, choices=["retrieval", "direct_score"],
        help="Set operating mode: 'retrieval' (search-based) or 'direct_score' (direct scoring)."
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
    parser.add_argument(
        "--no-use-serper-cache", action="store_true", default=False,
        help="Disable serper cache. When set, will not read from cache and always perform search."
    )
    parser.add_argument(
        "--num_threads", type=int, default=1,
        help="Number of threads for parallel processing. Set to 1 for single-threaded mode (default)."
    )
    return parser.parse_args()


# --- 工具定义 ---
SEARCH_TOOL_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "search_local",
            "description": "Searches the local Wikipedia document database. Use this for well-established factual knowledge, definitions, and historical information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query keyword for the local Wikipedia database."}},
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
                "properties": {"query": {"type": "string", "description": "The search query keyword for the web search engine."}},
                "required": ["query"],
            },
        },
    }
]


# --- 辅助函数 ---
def extract_final_scoring(model_generated_output: str) -> int:
    """从模型输出中提取最终分数。"""
    if not isinstance(model_generated_output, str):
        return 0
    match = re.search(r"<verdict>\s*(\d+)\s*</verdict>", model_generated_output, re.IGNORECASE)
    if not match:
        match = re.search(r"\*\*Final Verdict\*\*:\s*(\d+)\s*", model_generated_output, re.IGNORECASE)
    if not match:
        match = re.search(r"Final Verdict:\s*(\d+)\s*", model_generated_output, re.IGNORECASE)
    try:
        return int(match.group(1))
    except:
        return 0


def check_ready_for_evaluation(model_generated_output: str) -> bool:
    """检查模型是否准备好进入最终评估阶段。"""
    return "READY_FOR_EVALUATION" in model_generated_output or "READY_FOR_ANSWERING" in model_generated_output


def call_sglang_api(sglang_url: str, messages: List[Dict], tokenizer, args, tools=None):
    """
    通过 HTTP API 调用 SGLang 服务。
    使用 tokenizer 的 apply_chat_template 格式化消息（包括工具定义），确保模型看到的内容与原始代码一致。
    返回: (content, tool_calls, input_tokens, output_tokens)
    """
    formatted_prompt = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools,
        enable_thinking=(not args.disable_thinking)
    )
    
    # 计算输入token数
    input_tokens = len(tokenizer.encode(formatted_prompt, add_special_tokens=False))
    
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": formatted_prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_token,
    }
    
    if tools:
        payload["tools"] = tools
    
    # 发送请求
    try:
        response = requests.post(
            f"{sglang_url}/v1/chat/completions",
            json=payload,
            timeout=300
        )
        
        # 检查响应状态
        if response.status_code != 200:
            error_detail = ""
            try:
                error_detail = response.json()
            except:
                error_detail = response.text[:500]  # 只取前500字符
            logging.error(f"SGLang API request failed with status {response.status_code}: {error_detail}")
            logging.error(f"Request payload: {json.dumps(payload, ensure_ascii=False, indent=2)[:500]}")
            return "", [], input_tokens, 0
        
        response.raise_for_status()
        result = response.json()
        
        # 提取响应内容
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0]["message"]
            content = message.get("content", "") if isinstance(message, dict) else ""
            tool_calls = message.get("tool_calls", []) if isinstance(message, dict) else []
            
            # 计算输出 token 数：文本 content + 工具调用（模型生成工具调用也要算 token/费用）
            output_tokens = len(tokenizer.encode(content, add_special_tokens=False)) if content else 0
            if tool_calls:
                tool_calls_str = json.dumps(tool_calls, ensure_ascii=False)
                output_tokens += len(tokenizer.encode(tool_calls_str, add_special_tokens=False))

            return content, tool_calls, input_tokens, output_tokens
        else:
            logging.error(f"Unexpected API response: {result}")
            return "", [], input_tokens, 0
    except requests.RequestException as e:
        logging.error(f"SGLang API request failed: {e}")
        logging.error(f"Request URL: {sglang_url}/v1/chat/completions")
        return "", [], input_tokens, 0


# --- 智能体状态处理函数 ---
def process_decision_stage(messages, sglang_url, tokenizer, args):
    """
    决定下一步行动（检索或评估）。
    返回: (content, tool_calls, should_evaluate, input_tokens, output_tokens)
    """
    content, tool_calls, input_tokens, output_tokens = call_sglang_api(
        sglang_url,
        messages,
        tokenizer,
        args,
        tools=SEARCH_TOOL_DEFINITION
    )

    if content is None:
        content = ""
    
    # 构建 assistant 消息，包含 content 和 tool_calls（如果有）
    assistant_message = {"role": "assistant", "content": content}
    if tool_calls and len(tool_calls) > 0:
        assistant_message["tool_calls"] = tool_calls
    messages.append(assistant_message)
    
    if check_ready_for_evaluation(content):
        return content, tool_calls, True, input_tokens, output_tokens
    
    # 检查是否有工具调用
    if tool_calls and len(tool_calls) > 0:
        return content, tool_calls, False, input_tokens, output_tokens
    else:
        return content, tool_calls, True, input_tokens, output_tokens


def process_tool_execution_stage(tool_calls, messages, local_api, serper_api, tokenizer):
    """执行工具调用，添加结果到历史记录。"""
    if not tool_calls:
        return
    
    call = tool_calls[0]

    # 如果 call 是字符串，尝试解析
    if isinstance(call, str):
        try:
            call = json.loads(call)
        except json.JSONDecodeError:
            logging.warning(
                "Tool call returned but string could not be parsed as JSON: %s",
                call[:300] if call else "(empty)",
            )
            return

    try:
        # 解析工具调用信息
        if isinstance(call, dict):
            if "function" in call and isinstance(call["function"], dict):
                call_name = call["function"].get("name", "")
                params_raw = call["function"].get("arguments", "{}")
            else:
                call_name = call.get("name", "")
                params_raw = call.get("parameters", {})
        else:
            call_name = getattr(call, "name", "")
            params_raw = getattr(call, "parameters", {})
        
        # 解析参数为字典
        if isinstance(params_raw, str):
            try:
                params = json.loads(params_raw)
            except json.JSONDecodeError:
                params = {}
        elif isinstance(params_raw, dict):
            params = params_raw
        else:
            params = {}
        
        # 确保 params 是字典类型（如果解析后仍然是字符串，再次尝试解析）
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {}
        if not isinstance(params, dict):
            params = {}
        
        query = params.get("query", "")
        if not query:
            logging.warning(
                "Tool call returned but could not parse query (missing or invalid). call=%s",
                json.dumps(call, ensure_ascii=False)[:400] if isinstance(call, dict) else str(call)[:400],
            )
            return
        
        if call_name == "search_local":
            results = local_api.search_api_call([query])
            content = f"[Source: Local Wikipedia]\n{results[0]}" if results else "[Source: Local Wikipedia]\nNo results found."
        elif call_name == "search_web":
            results = serper_api.search_api_call([query])
            content = f"[Source: Google Search]\n{results[0]}" if results else "[Source: Google Search]\nNo results found."
        else:
            content = f"Unknown tool: {call_name}"
        
        messages.append({"role": "tool", "content": content})
    except Exception as e:
        logging.warning(
            "Tool call returned but failed to parse or execute: %s. call=%s",
            e,
            json.dumps(tool_calls[0], ensure_ascii=False, default=str)[:400] if tool_calls else "[]",
        )


def process_evaluation_stage_scoring(item, messages, sglang_url, tokenizer, args):
    """处理最终评分阶段。返回 (t0, t_end, evaluation_input_tokens, evaluation_output_tokens, evaluation_messages)。"""
    scoring_verdict_prompt_retrieval = """Based on the preceding conversation, your task is to score the factuality of the given answer from a scale of 1-10.

Question: {question}
Answer: {answer}

Please first provide your explanation, and then state your final verdict in the format: '**Final Verdict**: <verdict> a score between 1-10 </verdict>'. Example: '**Final Verdict**: <verdict> 3 </verdict>'."""

    scoring_verdict_prompt_direct = """You are an expert fact-checking assistant. Your task is to score the factuality of the given answer from a scale of 1-10 based on your internal knowledge.

Question: {question}
Answer: {answer}

Please first provide your explanation, and then state your final verdict in the format: '**Final Verdict**: <verdict> a score between 1-10 </verdict>'. Example: '**Final Verdict**: <verdict> 3 </verdict>'."""

    answers = item["answers"]
    t0 = time.perf_counter()
    evaluation_input_tokens = 0
    evaluation_output_tokens = 0
    evaluation_messages = []  # 用于完整 inference_trace

    for answer in answers:
        if args.mode == "retrieval":
            prompt = scoring_verdict_prompt_retrieval.format(
                question=item["question"],
                answer=answer["answer"]
            )
            final_messages = messages + [{"role": "user", "content": prompt}]
        else:  # direct_score mode
            prompt = scoring_verdict_prompt_direct.format(
                question=item["question"],
                answer=answer["answer"]
            )
            final_messages = [{"role": "user", "content": prompt}]

        content, _, inp_tok, out_tok = call_sglang_api(sglang_url, final_messages, tokenizer, args)
        evaluation_input_tokens += inp_tok
        evaluation_output_tokens += out_tok
        answer["verdict_response"] = content
        answer["predicted_scoring"] = extract_final_scoring(content)
        evaluation_messages.append({"role": "user", "content": prompt})
        evaluation_messages.append({"role": "assistant", "content": content})

    t_end = time.perf_counter()

    return t0, t_end, evaluation_input_tokens, evaluation_output_tokens, evaluation_messages


def process_single_item(item, item_id, args, tokenizer, local_api, serper_api, print_lock=None):
    """
    处理单个样本的完整流程：检索（如果需要）-> 评估 -> 返回结果
    print_lock: 多线程模式下用于串行化打印的锁，避免输出交错。
    """
    def _print(*a, **k):
        if print_lock is not None:
            with print_lock:
                print(*a, **k)
        else:
            print(*a, **k)

    scoring_search_prompt = """Your task is to gather information to verify an answer to a question. Based on the question, think and identify what information you need and generate search queries using the available tools.

Question: {question}

Leverage both tools (`search_local` for Wikipedia, `search_web` for Google) and generate one search query per turn. If you have enough information, respond with "READY_FOR_EVALUATION"."""

    # 初始化结果（深拷贝以避免修改原始数据）
    result_item = copy.deepcopy(item)
    messages = []
    
    # 检索阶段统计
    retrieval_time_second = 0.0
    retrieval_iteration_num = 0
    retrieval_input_tokens = 0
    retrieval_output_tokens = 0
    
    # 如果是检索模式，执行智能体循环
    if args.mode == "retrieval":
        _print(f"--- Processing item {item_id}: Retrieval mode ---")
        messages = [{"role": "user", "content": scoring_search_prompt.format(question=item["question"])}]
        MAX_TURNS = 5
        
        retrieval_start_time = time.perf_counter()
        
        for turn in range(MAX_TURNS):
            retrieval_iteration_num = turn + 1
            _print(f"  Turn {retrieval_iteration_num}: Deciding next action...")
            content, tool_calls, should_evaluate, input_tokens, output_tokens = process_decision_stage(
                messages, args.sglang_url_retrieval, tokenizer, args
            )
            
            # 累计检索阶段的token数
            retrieval_input_tokens += input_tokens
            retrieval_output_tokens += output_tokens
            
            if should_evaluate:
                _print(f"  Ready for evaluation after {retrieval_iteration_num} turns.")
                break
            
            if tool_calls and len(tool_calls) > 0:
                _print(f"  Executing tool call...")
                process_tool_execution_stage(
                    tool_calls, messages, local_api, serper_api, tokenizer
                )
            else:
                _print(f"  No tool calls, moving to evaluation.")
                break
        
        retrieval_end_time = time.perf_counter()
        retrieval_time_second = round(retrieval_end_time - retrieval_start_time, 4)
        
        # retrieval_path：工具定义 + 检索对话
        result_item["retrieval_path"] = [{"tools": SEARCH_TOOL_DEFINITION}] + messages
    
    # 最终评估阶段
    _print(f"--- Processing item {item_id}: Final evaluation ---")
    evaluation_start_time, evaluation_end_time, evaluation_input_tokens, evaluation_output_tokens, evaluation_messages = process_evaluation_stage_scoring(
        result_item, messages, args.sglang_url_evaluation, tokenizer, args
    )

    evaluation_time_second = round(evaluation_end_time - evaluation_start_time, 4)

    # 各模式下每个 answer 已包含 predicted_scoring（即 predicted_score）

    # 记录统计信息
    result_item["retrieval_time_second"] = retrieval_time_second
    result_item["retrieval_iteration_num"] = retrieval_iteration_num
    result_item["evaluation_time_second"] = evaluation_time_second
    result_item["retrieval_input_tokens"] = retrieval_input_tokens
    result_item["retrieval_output_tokens"] = retrieval_output_tokens
    result_item["evaluation_input_tokens"] = evaluation_input_tokens
    result_item["evaluation_output_tokens"] = evaluation_output_tokens

    return result_item


# --- 主函数 ---
def main():
    """主执行函数。"""
    args = parse_args()
    if args.sglang_url is not None:
        args.sglang_url_retrieval = args.sglang_url
        args.sglang_url_evaluation = args.sglang_url

    print(f"Arguments: {args}")
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f if line.strip()][:50]

    # 初始化搜索 API（如果需要）
    local_api = None
    serper_api = None
    if args.mode == "retrieval":
        print("Initializing search APIs: Local (Wiki) and Serper (Google)...")
        local_api = SearchAPILocal()
        use_cache = not args.no_use_serper_cache
        serper_api = SearchAPISerper(use_cache=use_cache)

    # 处理模式选择
    num_threads = args.num_threads
    if num_threads <= 1:
        # 单线程模式
        print(f"--- Processing {len(input_data)} items (single-threaded) ---")
        final_results = []
        for i, item in enumerate(input_data):
            print(f"[{i+1}/{len(input_data)}] Processing item {i}...")
            result_item = process_single_item(
                item, i, args, tokenizer, local_api, serper_api
            )
            final_results.append(result_item)
            print(f"Item {i} completed.")
    else:
        # 多线程模式：使用打印锁避免输出交错
        print_lock = threading.Lock()
        print(f"--- Processing {len(input_data)} items (multi-threaded, {num_threads} threads) ---")
        final_results = [None] * len(input_data)  # 预分配列表以保持顺序
        
        def process_with_index(indexed_item):
            """包装函数，用于多线程处理。"""
            i, item = indexed_item
            try:
                with print_lock:
                    print(f"[Thread] Processing item {i}...")
                result_item = process_single_item(
                    item, i, args, tokenizer, local_api, serper_api, print_lock=print_lock
                )
                with print_lock:
                    print(f"[Thread] Item {i} completed.")
                return i, result_item
            except Exception as e:
                logging.error(f"Error processing item {i}: {e}")
                return i, None
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交所有任务
            futures = {
                executor.submit(process_with_index, (i, item)): i 
                for i, item in enumerate(input_data)
            }
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    i, result_item = future.result()
                    if result_item is not None:
                        final_results[i] = result_item
                except Exception as e:
                    idx = futures[future]
                    logging.error(f"Failed to get result for item {idx}: {e}")
        
        # 过滤掉 None 值（如果有失败的）
        final_results = [r for r in final_results if r is not None]

    # 保存结果
    print(f"All sequences processed. Saving results...")
    output_path = args.output_file
    with open(output_path, "w", encoding="utf-8") as f_out:
        for result in final_results:
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Results saved to {output_path}")

    print("Scoring mode finished. Verdict step completed.")

    # 打印最终摘要
    print(f"\nSummary:\nDataset: {os.path.basename(args.input_file)}\n"
          f"Tokenizer: {os.path.basename(args.tokenizer_path)}\n"
          f"SGLang URL (retrieval): {args.sglang_url_retrieval}\n"
          f"SGLang URL (evaluation): {args.sglang_url_evaluation}\n"
          f"Mode: {args.mode}\n"
          f"Threads: {num_threads}\n"
          f"Total items processed: {len(final_results)}")


if __name__ == "__main__":
    main()
