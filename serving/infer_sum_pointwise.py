#!/usr/bin/env python3
"""
Pointwise summary/scoring script (Serving / API version).
Reads JSONL with retrieval_path (e.g. from infer_scoring.py), calls SGLang via HTTP API
to score each answer (1-10) and writes summary_time_second, summary_input_tokens, summary_output_tokens per answer.
"""
import argparse
import json
import logging
import os
import re
import time
from typing import List, Dict, Optional, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from transformers import AutoTokenizer


# --- 命令行参数 ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="Pointwise fact-checking verdict script using SGLang HTTP API."
    )
    parser.add_argument(
        "--sglang_url", type=str, default="http://localhost:30000",
        help="URL of the SGLang HTTP server."
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True,
        help="Path to the tokenizer (for prompt formatting)."
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Path to the input JSONL file (e.g. output from infer_scoring.py with retrieval_path)."
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Path to save the output JSONL file with verdicts."
    )
    parser.add_argument(
        "--disable_thinking", action="store_true", default=False,
        help="Disable the model's thinking process."
    )
    parser.add_argument(
        "--max_token", type=int, default=2048,
        help="Maximum new tokens to generate for verdicts."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature for verdict generation."
    )
    parser.add_argument(
        "--num_threads", type=int, default=1,
        help="Number of threads for parallel processing. Default 1 (single-threaded)."
    )
    return parser.parse_args()


# --- 解析与辅助 ---
def extract_final_scoring(model_generated_output: str) -> int:
    """从模型输出中提取最终分数（1-10）。"""
    if not isinstance(model_generated_output, str):
        return 0
    match = re.search(r"<verdict>\s*(\d+)\s*</verdict>", model_generated_output, re.IGNORECASE)
    if not match:
        match = re.search(r"\*\*Final Verdict\*\*:\s*(\d+)\s*", model_generated_output, re.IGNORECASE)
    if not match:
        match = re.search(r"Final Verdict:\s*(\d+)\s*", model_generated_output, re.IGNORECASE)
    try:
        return int(match.group(1))
    except Exception:
        return 0


def format_conversation_history(retrieval_path: List[Dict]) -> str:
    """将 retrieval_path 格式化为可读的 Search Turn / Search Result 文本，与 src 一致。若首项含 \"tools\" 则从 retrieval_path[2:] 遍历。"""
    formatted_parts = []
    turn_counter = 1

    if not retrieval_path:
        return ""

    if isinstance(retrieval_path[0], dict) and "tools" in retrieval_path[0]:
        messages_to_format = retrieval_path[2:]
    else:
        messages_to_format = retrieval_path[1:]

    for message in messages_to_format:
        role = message.get("role")
        content = message.get("content", "")

        if role == "assistant":
            formatted_parts.append(f"--- Search Turn {turn_counter} ---")
            formatted_parts.append(f"[Model's Thought Process and Action]\n{content.strip()}")
            turn_counter += 1
        elif role == "tool":
            formatted_parts.append(f"[Search Result]\n{content.strip()}")

    return "\n\n".join(formatted_parts)


# --- API 调用 ---
def call_sglang_api(sglang_url: str, messages: List[Dict], tokenizer, args) -> Tuple[str, int, int]:
    """
    通过 HTTP API 调用 SGLang，仅用于 verdict 生成（不传 tools）。
    返回: (content, input_tokens, output_tokens)
    """
    formatted_prompt = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=(not args.disable_thinking),
    )
    input_tokens = len(tokenizer.encode(formatted_prompt, add_special_tokens=False))

    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": formatted_prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_token,
    }

    try:
        response = requests.post(
            f"{sglang_url}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        if response.status_code != 200:
            try:
                err = response.json()
            except Exception:
                err = response.text[:500]
            logging.error(f"SGLang API failed: {response.status_code} {err}")
            return "", input_tokens, 0

        result = response.json()
        if "choices" not in result or not result["choices"]:
            logging.error(f"Unexpected API response: {result}")
            return "", input_tokens, 0

        message = result["choices"][0].get("message", {})
        content = message.get("content", "") or ""
        output_tokens = len(tokenizer.encode(content, add_special_tokens=False))
        return content, input_tokens, output_tokens

    except requests.RequestException as e:
        logging.error(f"SGLang API request failed: {e}")
        return "", input_tokens, 0


# --- 提示模板（评分，与 infer_scoring 一致）---
SUMMARY_SCORING_PROMPT = """Based on the preceding search history, your task is to score the factuality of the given answer from a scale of 1-10.

### SEARCH HISTORY ###
{search_history}

### VERIFICATION TASK ###
Question: {question}
Answer: {answer}

Please first provide your explanation, and then state your final verdict in the format: '**Final Verdict**: <verdict> a score between 1-10 </verdict>'. Example: '**Final Verdict**: <verdict> 3 </verdict>'."""


def process_single_item(
    item_index: int,
    input_data: List[Dict],
    args,
    tokenizer,
    print_lock: Optional[threading.Lock] = None,
) -> None:
    """处理单条样本：对该 item 下每个 answer 调用 API 做评分，写回 verdict_response、predicted_scoring、summary_*。"""
    def _print(*a, **k):
        if print_lock:
            with print_lock:
                print(*a, **k)
        else:
            print(*a, **k)

    item = input_data[item_index]
    retrieval_path = item.get("retrieval_path", [])
    question = item.get("question", "")
    answers = item.get("answers", [])

    for j, answer_obj in enumerate(answers):
        answer_text = answer_obj.get("answer", "")
        search_history = format_conversation_history(retrieval_path)
        prompt_content = SUMMARY_SCORING_PROMPT.format(
            search_history=search_history,
            question=question,
            answer=answer_text,
        )
        final_messages = [{"role": "user", "content": prompt_content}]

        t0 = time.perf_counter()
        content, inp_tok, out_tok = call_sglang_api(args.sglang_url, final_messages, tokenizer, args)
        elapsed = time.perf_counter() - t0

        answer_obj["verdict_response"] = content
        answer_obj["predicted_scoring"] = extract_final_scoring(content)
        answer_obj["summary_time_second"] = round(elapsed, 4)
        answer_obj["summary_input_tokens"] = inp_tok
        answer_obj["summary_output_tokens"] = out_tok


# --- 主函数 ---
def main():
    args = parse_args()

    print(f"Arguments: {args}")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f if line.strip()]

    n_items = len(input_data)
    if n_items == 0:
        print("No items found.")
        return

    num_threads = args.num_threads
    print_lock = threading.Lock() if num_threads > 1 else None

    if num_threads <= 1:
        print(f"--- Processing {n_items} items (single-threaded) ---")
        for i in range(n_items):
            print(f"[{i+1}/{n_items}] Processing item {i}...")
            process_single_item(i, input_data, args, tokenizer, print_lock)
            print(f"Item {i} completed.")
    else:
        print(f"--- Processing {n_items} items (multi-threaded, {num_threads} threads) ---")

        def process_with_index(idx):
            try:
                with print_lock:
                    print(f"[Thread] Processing item {idx}...")
                process_single_item(idx, input_data, args, tokenizer, print_lock)
                with print_lock:
                    print(f"[Thread] Item {idx} completed.")
                return idx, True
            except Exception as e:
                logging.error(f"Error processing item {idx}: {e}")
                return idx, False

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(process_with_index, i): i for i in range(n_items)}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    idx = futures[future]
                    logging.error(f"Failed to get result for item {idx}: {e}")

    print("All sequences processed. Saving results...")
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for result in input_data:
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Results saved to {args.output_file}")
    print("Summary step finished.")

    print(
        f"\nSummary:\nDataset: {os.path.basename(args.input_file)}\n"
        f"Tokenizer: {os.path.basename(args.tokenizer_path)}\n"
        f"SGLang URL: {args.sglang_url}\n"
        f"Threads: {num_threads}\n"
        f"Total items processed: {n_items}"
    )


if __name__ == "__main__":
    main()
