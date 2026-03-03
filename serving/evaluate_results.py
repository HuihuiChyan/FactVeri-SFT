#!/usr/bin/env python3
"""
读取 infer_scoring / infer_batch_sglang 输出的 JSONL，完成两件事：
1. 计算排序相关指标：precision_at_1, average_kendall_tau, invalid_prediction_ratio
2. 计算整份数据上各类统计量的平均值（如 retrieval_time_second, evaluation_time_second 等）
"""
import argparse
import json
import logging
from typing import List, Dict, Any, Optional

from scipy.stats import kendalltau


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate scoring/ranking JSONL output.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL path.")
    parser.add_argument("--output_file", type=str, default=None, help="Optional: write metrics JSON here.")
    return parser.parse_args()


def get_predicted_ranking(item: Dict) -> Optional[List[int]]:
    """
    从单条结果中得到 predicted_ranking（1-based 答案编号列表）。
    若已有 predicted_ranking 则直接返回；否则由 answers[].predicted_scoring 按分数降序推出。
    """
    pred = item.get("predicted_ranking")
    if isinstance(pred, list) and pred:
        return pred

    answers = item.get("answers") or []
    if not answers:
        return None

    indexed = [(i + 1, (a.get("predicted_scoring") if isinstance(a.get("predicted_scoring"), (int, float)) else 0)) for i, a in enumerate(answers)]
    indexed.sort(key=lambda x: -x[1])
    return [x[0] for x in indexed]


def evaluate_ranking_metrics(results: List[Dict]) -> Dict[str, float] | None:
    """
    计算与 infer_batch_sglang 中 evaluate_final_results_ranking 一致的三项指标：
    precision_at_1, average_kendall_tau, invalid_prediction_ratio
    """
    top_1_correct_count = 0
    valid_evaluation_count = 0
    invalid_predictions = 0
    total_items = len(results)
    kendall_tau_scores: List[float] = []

    for item in results:
        true_ranking = item.get("verify_result")
        pred_ranking = get_predicted_ranking(item)

        is_true_valid = isinstance(true_ranking, list) and len(true_ranking) > 0
        if not is_true_valid:
            continue

        is_pred_valid = isinstance(pred_ranking, list) and len(pred_ranking) > 0
        if not is_pred_valid or len(true_ranking) != len(pred_ranking):
            invalid_predictions += 1
            continue

        valid_evaluation_count += 1
        num_answers = len(true_ranking)

        if true_ranking[0] == pred_ranking[0]:
            top_1_correct_count += 1

        true_ranks = [0] * num_answers
        for rank, item_idx in enumerate(true_ranking):
            true_ranks[item_idx - 1] = rank

        pred_ranks = [0] * num_answers
        for rank, item_idx in enumerate(pred_ranking):
            pred_ranks[item_idx - 1] = rank

        tau, _ = kendalltau(true_ranks, pred_ranks)
        kendall_tau_scores.append(tau)

    if valid_evaluation_count == 0:
        logging.error("Evaluation failed. No valid items to evaluate.")
        return None

    precision_at_1 = top_1_correct_count / valid_evaluation_count
    avg_kendall_tau = sum(kendall_tau_scores) / len(kendall_tau_scores)
    invalid_ratio = invalid_predictions / total_items if total_items > 0 else 0.0

    return {
        "precision_at_1": round(precision_at_1, 4),
        "average_kendall_tau": round(avg_kendall_tau, 4),
        "invalid_prediction_ratio": round(invalid_ratio, 4),
    }


# 统计量：item 级 per question（infer_scoring）；summary 级 per response（infer_sum_pointwise 每个 answer 一个数）
ITEM_LEVEL_STATS = [
    "retrieval_time_second",
    "evaluation_time_second",
    "retrieval_input_tokens",
    "retrieval_output_tokens",
    "evaluation_input_tokens",
    "evaluation_output_tokens",
    "retrieval_iteration_num",
]
SUMMARY_PER_RESPONSE_STATS = [
    "summary_time_second",
    "summary_input_tokens",
    "summary_output_tokens",
]


def average_run_stats(results: List[Dict]) -> Dict[str, Any]:
    """
    item 级字段：per question，直接从 item 取数求平均。
    summary_* 字段：per response，对所有 answer 上的值求平均。
    """
    sums: Dict[str, float] = {k: 0.0 for k in ITEM_LEVEL_STATS}
    counts: Dict[str, int] = {k: 0 for k in ITEM_LEVEL_STATS}
    summary_sums: Dict[str, float] = {k: 0.0 for k in SUMMARY_PER_RESPONSE_STATS}
    summary_counts: Dict[str, int] = {k: 0 for k in SUMMARY_PER_RESPONSE_STATS}

    for item in results:
        for k in ITEM_LEVEL_STATS:
            v = item.get(k)
            if v is not None and isinstance(v, (int, float)):
                sums[k] += float(v)
                counts[k] += 1

        for answer in item.get("answers") or []:
            for k in SUMMARY_PER_RESPONSE_STATS:
                v = answer.get(k)
                if v is not None and isinstance(v, (int, float)):
                    summary_sums[k] += float(v)
                    summary_counts[k] += 1

    avg_stats = {}
    for k in ITEM_LEVEL_STATS:
        if counts[k] > 0:
            avg_stats[f"avg_{k}"] = round(sums[k] / counts[k], 4)
        else:
            avg_stats[f"avg_{k}"] = None
    for k in SUMMARY_PER_RESPONSE_STATS:
        if summary_counts[k] > 0:
            avg_stats[f"avg_{k}"] = round(summary_sums[k] / summary_counts[k], 4)
        else:
            avg_stats[f"avg_{k}"] = None
    return avg_stats


def main():
    args = parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]

    if not results:
        logging.error("No valid lines in input file.")
        return

    # 1) 排序三项指标
    ranking_metrics = evaluate_ranking_metrics(results)
    if ranking_metrics is not None:
        print("\n--- Ranking Evaluation Results ---")
        for key, value in ranking_metrics.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("----------------------------------\n")

    # 2) 整份数据上统计量平均
    avg_stats = average_run_stats(results)
    print("--- Dataset-wide Average Stats ---")
    for key, value in avg_stats.items():
        print(f"{key}: {value}")
    print("----------------------------------\n")

    out = {"ranking_metrics": ranking_metrics, "average_run_stats": avg_stats, "num_items": len(results)}
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Metrics written to {args.output_file}")


if __name__ == "__main__":
    main()
