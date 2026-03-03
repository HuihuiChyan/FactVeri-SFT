#!/usr/bin/env python3
"""
统计检索/推理结果 JSON/JSONL 中的各类统计量：

- 检索：每次检索时间、检索次数、检索 token 数（来自 infer_batch_sglang）
- 端到端：e2e_latency_seconds, total_sequence_tokens（若有）
- infer_sum_pointwise：每个 response 的耗时与 token（input/output/total）
- infer_classifier：每个 response 的耗时

支持 infer_batch_sglang 输出（含 retrieval_stats, total_retrieval_time_seconds,
total_retrieval_tokens）、infer_sum_pointwise 输出（answers[].infer_sum_pointwise_*）、
infer_classifier 输出（answers[].infer_classifier_time_seconds）。
"""

import argparse
import json
import os
import sys
from typing import List

# 允许从 src 导入评测函数（infer_batch_sglang / infer_sum_pointwise）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from infer_batch_sglang import evaluate_final_results_ranking
from infer_sum_pointwise import evaluate_final_results_pointwise


def get_metrics(obj: dict) -> dict | None:
    """
    从单条记录中提取统计字段。
    返回 dict: retrieval_time, retrieval_count, retrieval_tokens, e2e_latency_seconds,
    total_sequence_tokens；若完全无相关字段则返回 None。
    """
    # 检索时间与次数：顶层或 retrieval_stats
    time_s = obj.get("total_retrieval_time_seconds")
    count = obj.get("search_count")
    retrieval_tokens = obj.get("total_retrieval_tokens")
    stats = obj.get("retrieval_stats") or []
    if stats:
        if time_s is None:
            time_s = sum(s.get("time_seconds", 0) for s in stats)
        if count is None:
            count = len(stats)
        if retrieval_tokens is None:
            retrieval_tokens = sum(s.get("token_count", 0) for s in stats)

    # 从 answers 聚合（detach 等格式）
    if time_s is None and count is None and retrieval_tokens is None and obj.get("answers"):
        total_time = 0.0
        total_count = 0
        total_tokens = 0
        for ans in obj["answers"]:
            if isinstance(ans, dict):
                st = ans.get("retrieval_stats") or []
                total_time += sum(s.get("time_seconds", 0) for s in st)
                total_count += ans.get("search_count", len(st))
                total_tokens += sum(s.get("token_count", 0) for s in st)
        if total_count > 0 or total_time > 0 or total_tokens > 0:
            if time_s is None:
                time_s = total_time
            if count is None:
                count = total_count
            if retrieval_tokens is None:
                retrieval_tokens = total_tokens

    if count is None:
        count = 0
    if time_s is None:
        time_s = 0.0
    if retrieval_tokens is None:
        retrieval_tokens = 0

    # 端到端延迟与序列 token（新字段，旧文件可能没有）
    e2e = obj.get("e2e_latency_seconds")
    seq_tokens = obj.get("total_sequence_tokens")

    return {
        "retrieval_time": float(time_s),
        "retrieval_count": int(count),
        "retrieval_tokens": int(retrieval_tokens),
        "e2e_latency_seconds": e2e if e2e is not None else None,
        "total_sequence_tokens": int(seq_tokens) if seq_tokens is not None else None,
    }


def get_answer_level_metrics(obj: dict) -> List[dict]:
    """
    从单条记录中提取每条 answer 的 infer_sum_pointwise 与 infer_classifier 统计。
    返回 list of dict，每个元素对应一个 answer 的：
    infer_sum_pointwise_time_seconds, infer_sum_pointwise_input_tokens,
    infer_sum_pointwise_output_tokens, infer_sum_pointwise_total_tokens,
    infer_classifier_time_seconds。
    """
    out = []
    for ans in obj.get("answers") or []:
        if not isinstance(ans, dict):
            continue
        out.append({
            "infer_sum_pointwise_time_seconds": ans.get("infer_sum_pointwise_time_seconds"),
            "infer_sum_pointwise_input_tokens": ans.get("infer_sum_pointwise_input_tokens"),
            "infer_sum_pointwise_output_tokens": ans.get("infer_sum_pointwise_output_tokens"),
            "infer_sum_pointwise_total_tokens": ans.get("infer_sum_pointwise_total_tokens"),
            "infer_classifier_time_seconds": ans.get("infer_classifier_time_seconds"),
        })
    return out


def load_jsonl(path: str):
    """逐行读取 JSONL；若整文件是单个 JSON 数组则兼容解析。"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if text.startswith("["):
        try:
            data = json.loads(text)
            yield from data
            return
        except json.JSONDecodeError:
            pass
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skip invalid line: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="统计检索/推理结果 JSON/JSONL 的检索、infer_sum_pointwise、infer_classifier 等统计量。",
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="输入 JSON/JSONL 路径（可为 retrieval / sum / classifier 任一阶段输出）。",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="跳过无检索/无 e2e 等字段的样本（否则检索相关计为 0）",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="可选：将带统计量的结果写回 JSONL（保留原字段并确保含汇总统计）。",
    )
    args = parser.parse_args()

    retrieval_times = []
    retrieval_counts = []
    retrieval_tokens_list = []
    e2e_latencies = []
    sequence_tokens_list = []
    # 按 answer 维度的统计（用于 scoring 流程：sum_pointwise + classifier）
    sum_pointwise_times = []
    sum_pointwise_input_tokens = []
    sum_pointwise_output_tokens = []
    sum_pointwise_total_tokens = []
    classifier_times = []

    records = list(load_jsonl(args.input_file))
    records_for_output = records if args.output_file else []

    for obj in records:
        m = get_metrics(obj)
        if m is None:
            if not args.skip_missing:
                retrieval_times.append(0.0)
                retrieval_counts.append(0)
                retrieval_tokens_list.append(0)
                e2e_latencies.append(None)
                sequence_tokens_list.append(None)
            else:
                continue
        else:
            retrieval_times.append(m["retrieval_time"])
            retrieval_counts.append(m["retrieval_count"])
            retrieval_tokens_list.append(m["retrieval_tokens"])
            e2e_latencies.append(m["e2e_latency_seconds"])
            sequence_tokens_list.append(m["total_sequence_tokens"])

        # 每条 answer 的 sum_pointwise / classifier 统计
        for am in get_answer_level_metrics(obj):
            if am["infer_sum_pointwise_time_seconds"] is not None:
                sum_pointwise_times.append(float(am["infer_sum_pointwise_time_seconds"]))
            if am["infer_sum_pointwise_input_tokens"] is not None:
                sum_pointwise_input_tokens.append(int(am["infer_sum_pointwise_input_tokens"]))
            if am["infer_sum_pointwise_output_tokens"] is not None:
                sum_pointwise_output_tokens.append(int(am["infer_sum_pointwise_output_tokens"]))
            if am["infer_sum_pointwise_total_tokens"] is not None:
                sum_pointwise_total_tokens.append(int(am["infer_sum_pointwise_total_tokens"]))
            if am["infer_classifier_time_seconds"] is not None:
                classifier_times.append(float(am["infer_classifier_time_seconds"]))

    n = len(retrieval_times)
    if n == 0:
        print("No valid responses found.")
        return 1

    # 检索时间、次数、token：始终有（可默认为 0）
    avg_retrieval_time = sum(retrieval_times) / n
    avg_retrieval_count = sum(retrieval_counts) / n
    total_retrieval_time = sum(retrieval_times)
    total_retrieval_count = sum(retrieval_counts)
    avg_retrieval_tokens = sum(retrieval_tokens_list) / n if retrieval_tokens_list else 0
    total_retrieval_tokens = sum(retrieval_tokens_list)

    # 端到端与序列 token：仅统计有值的
    e2e_valid = [x for x in e2e_latencies if x is not None]
    seq_valid = [x for x in sequence_tokens_list if x is not None]

    n_answers = len(sum_pointwise_times) or len(classifier_times) or 1
    if sum_pointwise_times:
        n_answers = max(n_answers, len(sum_pointwise_times))
    if classifier_times:
        n_answers = max(n_answers, len(classifier_times))

    print(f"Input: {args.input_file}")
    print(f"Items (samples): {n}")
    if sum_pointwise_times or classifier_times:
        print(f"Answers (for sum/cls stats): {n_answers}")
    print()
    print("--- Retrieval (infer_batch_sglang) ---")
    print("Per-item averages:")
    print(f"  Retrieval time (s):       {avg_retrieval_time:.4f}")
    print(f"  Retrieval count:          {avg_retrieval_count:.2f}")
    print(f"  Retrieval tokens:         {avg_retrieval_tokens:.2f}")
    if e2e_valid:
        print(f"  E2E latency (s):          {sum(e2e_valid) / len(e2e_valid):.4f}  (n={len(e2e_valid)})")
    else:
        print(f"  E2E latency (s):          (no field in data)")
    if seq_valid:
        print(f"  Total sequence tokens:    {sum(seq_valid) / len(seq_valid):.2f}  (n={len(seq_valid)})")
    else:
        print(f"  Total sequence tokens:    (no field in data)")
    print("Totals:")
    print(f"  Total retrieval time (s):   {total_retrieval_time:.4f}")
    print(f"  Total retrieval count:     {total_retrieval_count}")
    print(f"  Total retrieval tokens:    {total_retrieval_tokens}")
    if e2e_valid:
        print(f"  Total E2E latency (s):     {sum(e2e_valid):.4f}")
    if seq_valid:
        print(f"  Total sequence tokens:    {sum(seq_valid)}")

    if sum_pointwise_times or sum_pointwise_input_tokens or sum_pointwise_output_tokens or sum_pointwise_total_tokens:
        print()
        print("--- infer_sum_pointwise (per-answer) ---")
        if sum_pointwise_times:
            print(f"  Time (s):     avg={sum(sum_pointwise_times)/len(sum_pointwise_times):.4f}, total={sum(sum_pointwise_times):.4f}, n={len(sum_pointwise_times)}")
        if sum_pointwise_input_tokens:
            print(f"  Input tokens:  avg={sum(sum_pointwise_input_tokens)/len(sum_pointwise_input_tokens):.2f}, total={sum(sum_pointwise_input_tokens)}, n={len(sum_pointwise_input_tokens)}")
        if sum_pointwise_output_tokens:
            print(f"  Output tokens: avg={sum(sum_pointwise_output_tokens)/len(sum_pointwise_output_tokens):.2f}, total={sum(sum_pointwise_output_tokens)}, n={len(sum_pointwise_output_tokens)}")
        if sum_pointwise_total_tokens:
            print(f"  Total tokens:  avg={sum(sum_pointwise_total_tokens)/len(sum_pointwise_total_tokens):.2f}, total={sum(sum_pointwise_total_tokens)}, n={len(sum_pointwise_total_tokens)}")

    if classifier_times:
        print()
        print("--- infer_classifier (per-answer) ---")
        print(f"  Time (s):     avg={sum(classifier_times)/len(classifier_times):.4f}, total={sum(classifier_times):.4f}, n={len(classifier_times)}")

    # Ranking 评测：P@1、Kendall's Tau 等（需含 predicted_ranking 与 verify_result）
    has_ranking = any(
        isinstance(obj.get("predicted_ranking"), list) and isinstance(obj.get("verify_result"), list)
        for obj in records
    )
    if has_ranking and records:
        evaluate_final_results_ranking(records)

    # Pointwise 评测：Accuracy、F1 等（需含 answers[].final_verdict）
    has_pointwise = any(
        any(isinstance(a.get("final_verdict"), str) for a in (obj.get("answers") or []) if isinstance(a, dict))
        for obj in records
    )
    if has_pointwise and records:
        evaluate_final_results_pointwise(records)

    if args.output_file and records_for_output:
        with open(args.output_file, "w", encoding="utf-8") as f_out:
            for rec in records_for_output:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print()
        print(f"Wrote {len(records_for_output)} records to {args.output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
