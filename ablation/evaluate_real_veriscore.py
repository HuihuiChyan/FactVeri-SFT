#!/usr/bin/env python3
"""
评测 run_real_veriscore.py 的输出文件
参照 compare_scoring.py 的逻辑，但适配 JSON 格式（而非 JSONL）
"""
import json
import sys
import logging
from typing import List, Dict, Union
from scipy.stats import kendalltau

# 这个函数是正确的，我们保留它，用于处理 'predicted_scoring'
def get_ranking_from_scores(scores: List[Union[int, float]]) -> List[int]:
    """
    根据分数列表生成排名。
    分数越高，排名越靠前（降序排序）。
    返回的是一个索引列表，表示排名顺序。
    
    例如: scores = [1.5, 3.1, 2.0]  (索引0, 1, 2的分数)
    返回: [1, 2, 0] 
    (表示: 索引1排第一, 索引2排第二, 索引0排第三)
    """
    
    # 将分数与它们的原始索引配对 (index, score)
    indexed_scores = list(enumerate(scores))
    
    # 根据分数（元组的第二个元素 item[1]）进行降序排序
    sorted_by_score = sorted(indexed_scores, key=lambda item: item[1], reverse=True)
    
    # 提取排序后的原始索引 (item[0])，这就是排名列表
    ranking = [item[0] for item in sorted_by_score]
    return ranking


def evaluate_final_results_ranking(results: List[Dict]):
    """
    计算并打印排名方案的评估指标。
    真实值是一个排名列表。使用的指标是 P@1 和 Kendall's Tau。
    """
    kendall_tau_scores = []
    top_1_correct_count, valid_evaluation_count, invalid_predictions = 0, 0, 0
    total_items = len(results)
    
    if total_items == 0:
        logging.warning("没有收到任何结果用于评估。")
        return None

    for item in results:
        # 注意：这里我们期望的 "verify_result" 和 "predicted_ranking" 
        # 是 *0-based 排名列表* (如 [0, 2, 1])
        true_ranking = item.get("verify_result")
        pred_ranking = item.get("predicted_ranking")
        
        is_true_label_valid = isinstance(true_ranking, list) and true_ranking
        if not is_true_label_valid: 
            continue

        is_pred_valid = isinstance(pred_ranking, list) and pred_ranking
        if not is_pred_valid or len(true_ranking) != len(pred_ranking):
            invalid_predictions += 1
            continue

        valid_evaluation_count += 1
        num_answers = len(true_ranking)

        # 检查 P@1 (排名第一的索引是否相同)
        if true_ranking[0] == pred_ranking[0]:
            top_1_correct_count += 1

        # --- 计算 Kendall's Tau ---
        # 我们需要将 *排名列表* (如 [0, 2, 1]) 转换为 *项目秩列表* (如 [0, 2, 1])
        # 排名列表: [idx_at_rank_0, idx_at_rank_1, idx_at_rank_2]
        # 项目秩列表: [rank_of_idx_0,   rank_of_idx_1,   rank_of_idx_2]

        true_ranks = [0] * num_answers
        for rank, item_idx in enumerate(true_ranking):
            if 0 <= item_idx < num_answers:
                true_ranks[item_idx] = rank
            else:
                logging.warning(f"在 true_ranking 中发现无效索引: {item_idx}")

        pred_ranks = [0] * num_answers
        for rank, item_idx in enumerate(pred_ranking):
            if 0 <= item_idx < num_answers:
                pred_ranks[item_idx] = rank
            else:
                logging.warning(f"在 pred_ranking 中发现无效索引: {item_idx}")

        tau, _ = kendalltau(true_ranks, pred_ranks)
        kendall_tau_scores.append(tau)

    if valid_evaluation_count == 0:
        logging.error("评估失败。没有有效的项目可供评估。")
        return None

    # --- 计算最终指标 ---
    precision_at_1 = top_1_correct_count / valid_evaluation_count if valid_evaluation_count else 0.0
    avg_kendall_tau = sum(kendall_tau_scores) / len(kendall_tau_scores) if kendall_tau_scores else 0.0
    invalid_ratio = invalid_predictions / total_items if total_items > 0 else 0.0

    metrics_dict = {
        "precision_at_1": round(precision_at_1, 4),
        "average_kendall_tau": round(avg_kendall_tau, 4),
        "invalid_prediction_ratio": round(invalid_ratio, 4),
        "valid_evaluation_count": valid_evaluation_count,
        "total_items_processed": total_items,
    }

    print("\n--- 📊 排名评估结果 ---")
    for key, value in metrics_dict.items(): 
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("----------------------------------\n")
    return metrics_dict


def analyze_rankings_from_file(filepath: str):
    """
    从 JSON 文件读取数据，生成排名，并调用评估函数。
    文件格式：JSON 数组，每个元素包含 question, answers, verify_result 等字段
    """
    print(f"正在开始分析文件: {filepath}...")
    
    results_for_evaluation: List[Dict] = [] # 用于存储所有排名的列表
    total_items = 0
    error_items = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"错误: 文件格式不正确，期望 JSON 数组，但得到 {type(data)}")
            return
        
        total_items = len(data)
        print(f"总共找到 {total_items} 个问题")
        
        for item_idx, item in enumerate(data):
            try:
                # 1. 提取 'predicted_scoring' 列表
                if 'answers' not in item or not isinstance(item['answers'], list) or not item['answers']:
                    logging.warning(f"跳过第 {item_idx + 1} 项: 'answers' 键缺失、不是列表或为空。")
                    error_items += 1
                    continue
                
                predicted_scores = [answer.get('veriscore_precision') for answer in item['answers']]
                
                if None in predicted_scores:
                    logging.warning(f"跳过第 {item_idx + 1} 项: 'answers' 列表中缺少 'veriscore_precision' 键。")
                    error_items += 1
                    continue

                # 2. 提取 'verify_result' (1-based 排名) 列表
                #    根据文件格式, 'verify_result' 是 [3, 1, 2] 这样的 1-based 排名列表
                if 'verify_result' not in item or not isinstance(item['verify_result'], list):
                    logging.warning(f"跳过第 {item_idx + 1} 项: 'verify_result' 键缺失或不是列表。")
                    error_items += 1
                    continue
                    
                verified_1_based_ranking = item['verify_result'] # 这是 1-based 排名, 如 [3, 1, 2]

                # 3. 检查列表长度是否一致
                if len(predicted_scores) != len(verified_1_based_ranking):
                    logging.warning(f"跳过第 {item_idx + 1} 项: 分数/排名列表长度不匹配 (分数: {len(predicted_scores)}, 排名: {len(verified_1_based_ranking)})。")
                    error_items += 1
                    continue

                # 4. 生成排名
                
                # (A) 预测排名: 从分数 -> 0-based 排名
                predicted_ranking_list = get_ranking_from_scores(predicted_scores)
                
                # (B) 真实排名: 从 1-based 排名 -> 0-based 排名
                try:
                    # e.g., [3, 1, 2] -> [2, 0, 1]
                    verified_ranking_list = [idx - 1 for idx in verified_1_based_ranking]
                    
                    # 检查转换后的索引是否有效
                    num_items = len(verified_ranking_list)
                    if not all(0 <= idx < num_items for idx in verified_ranking_list):
                         logging.warning(f"跳过第 {item_idx + 1} 项: 'verify_result' 包含无效的 1-based 索引 (例如 0 或 大于 {num_items})。")
                         error_items += 1
                         continue
                    # 检查是否有重复索引，这在排名中是不允许的
                    if len(set(verified_ranking_list)) != num_items:
                         logging.warning(f"跳过第 {item_idx + 1} 项: 'verify_result' 转换后包含重复的 0-based 索引。")
                         error_items += 1
                         continue

                except TypeError:
                    logging.warning(f"跳过第 {item_idx + 1} 项: 'verify_result' 包含非整数项，无法转换为 0-based 索引。")
                    error_items += 1
                    continue
                
                # 5. 将排名列表添加到我们的结果集中
                results_for_evaluation.append({
                    "verify_result": verified_ranking_list,      # 格式: [2, 0, 1]
                    "predicted_ranking": predicted_ranking_list  # 格式: [1, 2, 0]
                })

            except (KeyError, TypeError, AttributeError) as e:
                logging.error(f"跳过第 {item_idx + 1} 项: 处理数据时出错 - {e}")
                error_items += 1

        # --- 文件处理摘要 ---
        print("\n--- 📁 文件处理摘要 ---")
        print(f"总共检查项数: {total_items}")
        print(f"跳过/错误项数: {error_items}")
        valid_items = total_items - error_items
        print(f"有效参与评估项数: {valid_items}")

        if valid_items > 0:
            # 6. (循环结束后) 调用评估函数
            print("正在计算排名统计数据...")
            evaluate_final_results_ranking(results_for_evaluation)
        else:
            print("没有可用于评估的有效数据项。")

    except FileNotFoundError:
        print(f"错误: 文件 '{filepath}' 未找到。")
    except json.JSONDecodeError as e:
        print(f"错误: JSON 解析失败 - {e}")
    except Exception as e:
        print(f"发生意外错误: {e}")
        import traceback
        traceback.print_exc()

# --- 脚本主入口 ---
if __name__ == "__main__":
    # 配置日志记录
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s') 
    
    if len(sys.argv) < 2:
        print("用法: python evaluate_real_veriscore.py <your_file.json>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    analyze_rankings_from_file(file_path)
