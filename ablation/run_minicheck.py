import json
import os
import sys
import logging
from typing import List, Dict, Union
from scipy.stats import kendalltau

# from minicheck.minicheck import MiniCheck
from transformers import AutoModelForSequenceClassification

# --- 配置 ---
MODEL_NAME = 'Bespoke-MiniCheck-7B'
CACHE_DIR = '/workspace/HFModels'

def get_ranking_from_scores(scores: List[Union[int, float]]) -> List[int]:
    """
    根据分数列表生成排名。
    分数越高，排名越靠前 (reverse=True)。
    返回的是一个 0-based 索引列表，表示排名顺序。
    
    例如: scores = [0.8, 0.9, 0.5] (索引0, 1, 2的分数)
    返回: [1, 0, 2] 
    (表示: 索引1排第一, 索引0排第二, 索引2排第三)
    """
    indexed_scores = list(enumerate(scores))
    
    # *** 关键: reverse=True ***
    # 因为 raw_prob 是相似度得分，越高越好，所以我们降序排序。
    sorted_by_score = sorted(indexed_scores, key=lambda item: item[1], reverse=True)
    
    ranking = [item[0] for item in sorted_by_score]
    return ranking


def evaluate_final_results_ranking(results: List[Dict]):
    """
    计算并打印排名方案的评估指标 (P@1, Kendall's Tau)。
    (此函数来自您的示例，用于处理 0-based 排名列表)
    """
    kendall_tau_scores = []
    top_1_correct_count, valid_evaluation_count, invalid_predictions = 0, 0, 0
    total_items = len(results)
    
    if total_items == 0:
        logging.warning("没有收到任何结果用于评估。")
        return None

    for item in results:
        true_ranking = item.get("verify_result")     # 期望格式: [1, 0, 2]
        pred_ranking = item.get("predicted_ranking") # 期望格式: [1, 0, 2]
        
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
        # 将 *排名列表* (如 [1, 0, 2]) 转换为 *项目秩列表* (如 [1, 0, 2] -> [1, 0, 2])
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

        try:
            tau, _ = kendalltau(true_ranks, pred_ranks)
            kendall_tau_scores.append(tau)
        except ValueError as e:
            logging.warning(f"计算 Kendall's Tau 时出错: {e}。真实秩: {true_ranks}, 预测秩: {pred_ranks}")

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


# -------------------------------------------------------------------
# 3. 合并后的主处理函数
# -------------------------------------------------------------------

def process_and_evaluate_file(filepath: str, scorer):
    """
    从 .jsonl 文件读取数据，使用 MiniCheck 计算分数生成排名，并调用评估函数。
    """
    print(f"正在开始分析文件: {filepath}...")
    
    results_for_evaluation: List[Dict] = [] # 用于存储所有排名的列表
    total_lines = 0
    error_lines = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line_num = i + 1
                total_lines += 1
                
                try:
                    data = json.loads(line)

                    # 1. 提取 'answers' 列表
                    if 'answers' not in data or not isinstance(data['answers'], list) or not data['answers']:
                        logging.warning(f"跳过第 {line_num} 行: 'answers' 键缺失、不是列表或为空。")
                        error_lines += 1
                        continue
                    
                    answers_data = data['answers']
                    
                    # 2. 提取 'verify_result' (1-based 排名) 列表
                    if 'verify_result' not in data or not isinstance(data['verify_result'], list):
                        logging.warning(f"跳过第 {line_num} 行: 'verify_result' 键缺失或不是列表。")
                        error_lines += 1
                        continue
                    
                    verified_1_based_ranking = data['verify_result'] # 这是 1-based 排名, 如 [2, 1, 3]

                    # 3. 提取用于 MiniCheck 计分的数据
                    docs_list = []      # 'useful_facts'
                    claims_list = []    # 'answer'
                    
                    for item in answers_data:
                        useful_facts = item.get('useful_facts')
                        answer_text = item.get('answer')
                        if useful_facts is None or answer_text is None:
                            # 如果任一行为空，则此行无法评估
                            raise ValueError(f"第 {line_num} 行的 'answers' 列表中缺少 'useful_facts' 或 'answer'。")
                        docs_list.append(useful_facts)
                        claims_list.append(answer_text)

                    # 4. 检查列表长度是否一致
                    if len(claims_list) != len(verified_1_based_ranking):
                        logging.warning(f"跳过第 {line_num} 行: 'answers' 列表 ({len(claims_list)}) 与 'verify_result' ({len(verified_1_based_ranking)}) 长度不匹配。")
                        error_lines += 1
                        continue

                    # 5. 生成排名
                    
                    # (A) 预测排名: 
                    #    (i) 计算 MiniCheck 分数 (raw_prob)
                    # print(f"正在为第 {line_num} 行的 {len(claims_list)} 个答案计算得分...")
                    # _, raw_prob_list, _, _ = scorer.score(docs=docs_list, claims=claims_list)
                    pairs = [(doc, claim) for doc, claim in zip(docs_list, claims_list)]
                    raw_prob_list = scorer.predict(pairs)
                    
                    #    (ii) 从分数 -> 0-based 排名
                    predicted_ranking_list = get_ranking_from_scores(raw_prob_list)
                    
                    # (B) 真实排名: 从 1-based 排名 -> 0-based 排名
                    try:
                        # e.g., [2, 1, 3] -> [1, 0, 2]
                        verified_0_based_ranking = [idx - 1 for idx in verified_1_based_ranking]
                        
                        # 验证转换后的索引
                        num_items = len(verified_0_based_ranking)
                        if not all(0 <= idx < num_items for idx in verified_0_based_ranking):
                            logging.warning(f"跳过第 {line_num} 行: 'verify_result' 包含无效的 1-based 索引 (例如 0 或 大于 {num_items})。")
                            error_lines += 1
                            continue
                        if len(set(verified_0_based_ranking)) != num_items:
                            logging.warning(f"跳过第 {line_num} 行: 'verify_result' 转换后包含重复的 0-based 索引。")
                            error_lines += 1
                            continue
                    
                    except TypeError:
                        logging.warning(f"跳过第 {line_num} 行: 'verify_result' 包含非整数项。")
                        error_lines += 1
                        continue
                    
                    # 6. 将排名列表添加到我们的结果集中
                    results_for_evaluation.append({
                        "verify_result": verified_0_based_ranking,   # 格式: [1, 0, 2]
                        "predicted_ranking": predicted_ranking_list  # 格式: [1, 0, 2]
                    })

                except json.JSONDecodeError:
                    logging.error(f"跳过第 {line_num} 行: JSON 解析错误。")
                    error_lines += 1
                except (KeyError, TypeError, AttributeError, ValueError) as e:
                    logging.error(f"跳过第 {line_num} 行: 处理数据时出错 - {e}")
                    error_lines += 1

    except FileNotFoundError:
        print(f"错误: 文件 '{filepath}' 未找到。", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"处理文件时发生意外错误: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 文件处理摘要 ---
    print("\n--- 📁 文件处理摘要 ---")
    print(f"总共检查行数: {total_lines}")
    print(f"跳过/错误行数: {error_lines}")
    valid_lines = total_lines - error_lines
    print(f"有效参与评估行数: {valid_lines}")

    if valid_lines > 0:
        # 7. (循环结束后) 调用评估函数
        print("正在计算最终排名统计数据...")
        evaluate_final_results_ranking(results_for_evaluation)
    else:
        print("没有可用于评估的有效数据行。")


# --- 脚本主入口 ---
if __name__ == "__main__":
    # 配置日志记录
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s') 
    
    if len(sys.argv) < 2:
        print("用法: python rank_and_evaluate.py <your_file.jsonl>")
        print("示例: python rank_and_evaluate.py data.jsonl")
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    # 1. 初始化 Scorer (全局一次)
    # scorer = MiniCheck(model_name=MODEL_NAME, enable_prefix_caching=False, cache_dir=CACHE_DIR)
    # Step 1: Load the model
    scorer = AutoModelForSequenceClassification.from_pretrained(
    '/workspace/HFModels/hallucination_evaluation_model', trust_remote_code=True)
    
    # 2. 处理文件并评估
    process_and_evaluate_file(file_path, scorer)