import json
import os
import sys
import logging
from typing import List, Dict, Union
from scipy.stats import kendalltau

# --- 配置 ---
# !! 重要: 请修改为您存放 1.json, 2.json, 3.json 文件的目录
# 假设: <SCORE_FILES_DIR>/1.json, <SCORE_FILES_DIR>/2.json, ...
SCORE_FILES_DIR = '.' 

# (这部分配置通常不需要修改)
# 键 0 对应 位置0 (ans_index=0)
SCORE_FILENAMES = {
    0: '/workspace/FactVeri-SFT/corpora/factscore_veri/ChatGPT_select_factscore_output.json',
    1: '/workspace/FactVeri-SFT/corpora/factscore_veri/InstructGPT_select_factscore_output.json',
    2: '/workspace/FactVeri-SFT/corpora/factscore_veri/PerplexityAI_select_factscore_output.json',
}
SCORES_KEY = 'scores' # 1.json 等文件中的分数列表的键名
# --- 更改结束 ---


def get_ranking_from_scores(scores: List[Union[int, float]]) -> List[int]:
    """
    根据分数列表生成排名。
    分数越高，排名越靠前 (reverse=True)。
    """
    indexed_scores = list(enumerate(scores))
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
        true_ranking = item.get("verify_result")      # 期望格式: [1, 0, 2]
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
# (更改) 新的辅助函数：用于预加载所有分数
# -------------------------------------------------------------------

def load_all_scores(base_dir: str) -> Dict[int, List[float]]:
    """
    从 1.json, 2.json, 3.json 加载所有分数列表。
    返回一个字典: {0: [scores_list_pos0], 1: [scores_list_pos1], 2: [scores_list_pos2]}
    """
    logging.info("正在预加载所有分数文件...")
    all_scores = {}
    expected_length = -1
    
    # 确保按顺序 0, 1, 2 加载
    for position in sorted(SCORE_FILENAMES.keys()):
        filename = SCORE_FILENAMES[position]
        filepath = filename
        
        # try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if SCORES_KEY not in data:
            logging.error(f"错误: 分数文件 '{filepath}' 中缺少 '{SCORES_KEY}' 键。")
            sys.exit(1)
        
        scores_list = data[SCORES_KEY]
        
        if not isinstance(scores_list, list):
            logging.error(f"错误: '{filepath}' 中的 '{SCORES_KEY}' 不是一个列表。")
            sys.exit(1)

        all_scores[position] = scores_list
        current_length = len(scores_list)
        
        # 验证所有分数列表的长度是否一致
        if expected_length == -1:
            expected_length = current_length
        elif expected_length != current_length:
            logging.error(f"错误: 分数文件 '{filename}' 的长度 ({current_length}) 与 '1.json' 的长度 ({expected_length}) 不匹配。")
            logging.error("错误: 1.json, 2.json, 3.json 中的 'scores' 列表长度必须完全一致。")
            sys.exit(1)

        # except FileNotFoundError:
        #     logging.error(f"错误: 找不到必要的分数文件: {filepath}")
        #     sys.exit(1)
        # except json.JSONDecodeError:
        #     logging.error(f"错误: 解析分数文件 '{filepath}' (JSON) 时出错。")
        #     sys.exit(1)
    
    if len(all_scores) != 3:
        logging.error("错误: 未能成功加载所有 3 个分数文件。")
        sys.exit(1)
        
    logging.info(f"✅ 成功加载 3 个分数列表，每个列表包含 {expected_length} 个分数。")
    return all_scores

# -------------------------------------------------------------------
# 3. 合并后的主处理函数
# -------------------------------------------------------------------

# --- 更改: 添加了 'all_scores' 参数 ---
def process_and_evaluate_file(filepath: str, all_scores: Dict[int, List[float]]):
    """
    从 .jsonl 文件读取数据，使用预加载的分数生成排名，并调用评估函数。
    """
    print(f"正在开始分析文件: {filepath}...")
    
    results_for_evaluation: List[Dict] = [] # 用于存储所有排名的列表
    total_lines = 0
    error_lines = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # --- 'i' 是 0-based 行索引 ---
            for i, line in enumerate(f):
                line_num = i + 1
                total_lines += 1
                
                try:
                    data = json.loads(line)

                    # 1. 提取 'answers' 列表
                    if 'answers' not in data or not isinstance(data['answers'], list):
                        logging.warning(f"跳过第 {line_num} 行: 'answers' 键缺失或不是列表。")
                        error_lines += 1
                        continue
                    
                    answers_data = data['answers']
                    
                    # 2. 提取 'verify_result' (1-based 排名) 列表
                    if 'verify_result' not in data or not isinstance(data['verify_result'], list):
                        logging.warning(f"跳过第 {line_num} 行: 'verify_result' 键缺失或不是列表。")
                        error_lines += 1
                        continue
                    
                    verified_1_based_ranking = data['verify_result'] 

                    # 3. (更改) 严格检查每行是否恰好有 3 个回答
                    if len(answers_data) != 3 or len(verified_1_based_ranking) != 3:
                        logging.warning(f"跳过第 {line_num} 行: 该行不包含 3 个回答。 'answers' 长度: {len(answers_data)}, 'verify_result' 长度: {len(verified_1_based_ranking)}")
                        error_lines += 1
                        continue

                    # 4. (更改) 生成排名
                    
                    # (A) 预测排名: 
                    # 
                    # --- 更改开始: 从预加载的列表 'all_scores' 中获取分数 ---
                    raw_prob_list = []
                    try:
                        # 'i' 是 0-based 行索引
                        score_pos_0 = all_scores[0][i] # 1.json[i]
                        score_pos_1 = all_scores[1][i] # 2.json[i]
                        score_pos_2 = all_scores[2][i] # 3.json[i]
                        raw_prob_list = [score_pos_0, score_pos_1, score_pos_2]
                    
                    except IndexError:
                        # 当 .jsonl 文件行数 > 分数列表长度时触发
                        logging.error(f"跳过第 {line_num} 行: 索引 {i} 超出了分数列表的范围 (长度 {len(all_scores[0])})。")
                        logging.error("请检查您的 .jsonl 文件和 1.json/2.json/3.json 文件是否匹配。")
                        error_lines += 1
                        continue # 跳过当前 .jsonl 行
                    # --- 更改结束 ---

                    
                    # (ii) 从分数 -> 0-based 排名
                    predicted_ranking_list = get_ranking_from_scores(raw_prob_list)
                    
                    # (B) 真实排名: 从 1-based 排名 -> 0-based 排名
                    try:
                        # e.g., [2, 1, 3] -> [1, 0, 2]
                        verified_0_based_ranking = [idx - 1 for idx in verified_1_based_ranking]
                        
                        # 验证转换后的索引
                        num_items = len(verified_0_based_ranking) # 应该总是 3
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
                    
                    # 5. 将排名列表添加到我们的结果集中
                    results_for_evaluation.append({
                        "verify_result": verified_0_based_ranking,    # 格式: [1, 0, 2]
                        "predicted_ranking": predicted_ranking_list   # 格式: [1, 0, 2]
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
        # 6. (循环结束后) 调用评估函数
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
    
    # 1. (更改) 预加载所有分数
    # 检查 SCORE_FILES_DIR 是否存在
    if not os.path.isdir(SCORE_FILES_DIR):
        logging.error(f"错误: 配置的分数目录 'SCORE_FILES_DIR' 不存在: {SCORE_FILES_DIR}")
        logging.error("请在脚本顶部创建此目录或修改 'SCORE_FILES_DIR' 变量。")
        sys.exit(1)
    
    all_scores = load_all_scores(SCORE_FILES_DIR)
    
    # 2. (更改) 健全性检查：比较 .jsonl 行数和分数列表长度
    try:
        score_list_length = len(all_scores[0])
        jsonl_line_count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                jsonl_line_count += 1
        
        if jsonl_line_count != score_list_length:
            logging.warning(f"!! 警告: 文件 '{file_path}' 有 {jsonl_line_count} 行,")
            logging.warning(f"   但分数文件 (1.json 等) 包含 {score_list_length} 个分数。")
            logging.warning("   将继续处理，但如果行数不匹配，评估可能不准确或在中途失败。")
        else:
            logging.info(f"✅ 文件行数 ({jsonl_line_count}) 与分数列表长度 ({score_list_length}) 匹配。")

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"检查文件行数时出错: {e}", file=sys.stderr)
        sys.exit(1)

    
    # 3. (更改) 处理文件并评估
    process_and_evaluate_file(file_path, all_scores)