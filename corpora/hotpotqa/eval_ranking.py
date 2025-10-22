#!/usr/bin/env python3

import json
import argparse
import sys
from scipy.stats import kendalltau

def calculate_precision_at_1(verify_result, predicted_ranking):
    """
    计算 Precision@1 (P@1).
    如果两个列表的第一个元素相同，则返回 1，否则返回 0。
    """
    # 检查列表是否为空
    if not verify_result or not predicted_ranking:
        return 0
    
    # 比较第一个元素
    return 1 if verify_result[0] == predicted_ranking[0] else 0

def calculate_kendall_tau(verify_result, predicted_ranking):
    """
    计算两个排名列表的 Kendall's Tau 系数。
    只有当两个列表包含完全相同的 *元素集* 时，计算才有意义。
    返回 tau 系数，如果无法计算则返回 None。
    """
    # 检查列表是否为空
    if not verify_result or not predicted_ranking:
        return None
    
    # Kendall's Tau 必须在两个包含相同元素的列表上计算。
    # 我们使用 set() 来检查。
    if set(verify_result) != set(predicted_ranking):
        # 如果集合不同，Tau 没有意义
        return None
    
    # 理论上，如果集合相同，长度也应该相同，但为了安全起见再次检查。
    if len(verify_result) != len(predicted_ranking):
         return None 

    try:
        # 为了计算 tau，我们需要比较 "ground truth" 顺序和 "predicted" 顺序。
        # 我们创建一个 map，用于查找 "predicted" 列表中的 (item -> rank)。
        predicted_rank_map = {item: rank for rank, item in enumerate(predicted_ranking)}
        
        # 'verify_ranks' 是 ground truth 的顺序 [0, 1, 2, ...]
        verify_ranks = list(range(len(verify_result)))
        
        # 'predicted_ranks_for_verify_items' 是 "predicted" 列表中
        # 对应 "verify_result" 列表顺序的 *排名* 列表。
        
        # 示例:
        # verify_result = [0, 1, 2] (0=rank 0, 1=rank 1, 2=rank 2)
        # predicted_ranking = [2, 0, 1] (2=rank 0, 0=rank 1, 1=rank 2)
        # predicted_rank_map = {2: 0, 0: 1, 1: 2}
        #
        # 我们最终比较的是 [0, 1, 2] 和 [1, 2, 0]
        
        predicted_ranks_for_verify_items = [predicted_rank_map[item] for item in verify_result]
        
        # 计算 Kendall's Tau
        tau, p_value = kendalltau(verify_ranks, predicted_ranks_for_verify_items)
        return tau

    except TypeError:
        # 如果列表中的项不可哈希（例如，它们是字典或列表），则会发生此错误
        print(f"警告: 行中的项不可哈希 (例如，列表/字典)。跳过该行的 Tau 计算。", file=sys.stderr)
        return None
    except Exception as e:
        print(f"警告: 计算 Kendall's Tau 时出错: {e}。跳过该行。", file=sys.stderr)
        return None

def process_jsonl(file_path):
    """
    读取 JSONL 文件并计算 P@1 和 Kendall's Tau。
    """
    all_p1_scores = []
    all_tau_scores = []
    total_lines = 0
    processed_lines = 0
    tau_valid_lines = 0
    
    print(f"开始处理文件: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                total_lines += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"警告: 第 {line_number} 行 JSON 格式无效，已跳过。", file=sys.stderr)
                    continue

                verify_result = data.get("verify_result")
                # verify_result = [v-1 for v in verify_result]
                if "predicted_ranking" in data.keys():
                    predicted_ranking = data.get("predicted_ranking")
                else:
                    predicted_ranking = data.get("final_verdict")

                # 检查 key 是否存在且值为列表
                if isinstance(verify_result, list) and isinstance(predicted_ranking, list):
                    processed_lines += 1
                    
                    # 1. 计算 Precision@1
                    p1 = calculate_precision_at_1(verify_result, predicted_ranking)
                    all_p1_scores.append(p1)
                    
                    # 2. 计算 Kendall's Tau
                    tau = calculate_kendall_tau(verify_result, predicted_ranking)
                    if tau is not None:
                        all_tau_scores.append(tau)
                        tau_valid_lines += 1
                    else:
                        # 仅在 P@1 有效但 Tau 无效时打印警告
                        print(f"提示: 第 {line_number} 行的 'verify_result' 和 'predicted_ranking' 列表元素集合不同。跳过该行的 Tau 计算。", file=sys.stderr)

                else:
                    print(f"警告: 第 {line_number} 行缺少 'verify_result' 或 'predicted_ranking' 字段，或字段不是列表。已跳过。", file=sys.stderr)

    except FileNotFoundError:
        print(f"错误: 文件未找到: {file_path}", file=sys.stderr)
        return
    except Exception as e:
        print(f"处理文件时发生意外错误: {e}", file=sys.stderr)
        return

    # --- 计算并打印结果 ---
    print("\n" + "="*30)
    print("         评估结果")
    print("="*30)
    print(f"总共读取行数: {total_lines}")
    print(f"成功处理行数 (P@1): {processed_lines}")
    print(f"Tau 有效行数 (元素集相同): {tau_valid_lines}")

    # P@1 结果
    if processed_lines > 0:
        avg_p1 = (sum(all_p1_scores) / processed_lines) * 100
        print(f"\n平均 Precision@1: {avg_p1:.2f}%")
    else:
        print("\n平均 Precision@1: N/A (没有可处理的数据)")

    # Kendall's Tau 结果
    if tau_valid_lines > 0:
        avg_tau = sum(all_tau_scores) / tau_valid_lines
        print(f"平均 Kendall's Tau: {avg_tau:.4f}")
    else:
        print(f"平均 Kendall's Tau: N/A (没有 Tau 有效的行)")

def main():
    parser = argparse.ArgumentParser(description="从 JSONL 文件计算 P@1 和 Kendall's Tau。")
    parser.add_argument("--jsonl_file", help="输入的 .jsonl 文件路径", default="hotpotqa_verification-Qwen2.5-7B-Instruct-retrieval-pointwise-only_facts-cls.json")
    args = parser.parse_args()
    
    process_jsonl(args.jsonl_file)

if __name__ == "__main__":
    main()