# Copyright 2024 Bytedance Ltd. and/or its affiliates
# ( ... Apache License 2.0 ... )

import re
import string
import random
import json
import sys
from collections import Counter  # 导入 Counter，用于计算F1

# --- 核心函数 (normalize_answer, em_check) (保留) ---

def normalize_answer(s):
    """小写、移除标点、文章和多余空格"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    """检查预测是否与任何一个标准答案完全匹配 (经过标准化处理)"""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    
    if prediction is None:
        prediction = ""
        
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        if golden_answer is None:
            golden_answer = ""
            
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score

# --- 新增：F1 Score 计算函数 ---

def compute_f1(prediction, ground_truth):
    """计算单个预测和单个标准答案之间的F1 score"""
    # 标准化
    prediction_normalized = normalize_answer(prediction)
    ground_truth_normalized = normalize_answer(ground_truth)
    
    # 分词
    prediction_tokens = prediction_normalized.split()
    ground_truth_tokens = ground_truth_normalized.split()
    
    # 处理空字符串的边缘情况
    if not prediction_tokens and not ground_truth_tokens:
        return 1.0  # 两个都是空的，算完全匹配
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0  # 一个是空的，一个是空的，F1为0

    # 使用 Counter 计算交集
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0

    # 计算 Precision, Recall, F1
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def f1_check(prediction, golden_answers):
    """
    计算预测与所有标准答案的F1 score，并返回最高分
    (QA评测标准：max F1 over all ground truths)
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    
    if prediction is None:
        prediction = ""

    # 如果 golden_answers 列表为空，无法比较
    if not golden_answers:
        return 0.0 # 或者根据需要抛出错误

    # 计算与每一个 golden answer 的F1，取最大值
    max_f1 = 0.0
    for golden_answer in golden_answers:
        if golden_answer is None:
            golden_answer = ""
        
        f1 = compute_f1(prediction, golden_answer)
        if f1 > max_f1:
            max_f1 = f1
            
    return max_f1

# --- 保留的其他函数 (subem_check, extract_solution, etc.) ---
# (这部分在本次计算中不会被调用)
def subem_check(prediction, golden_answers):
    # ... (代码与你提供的一致, 省略) ...
    pass

def extract_solution(solution_str):
    # ... (代码与你提供的一致, 省略) ...
    pass

def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    # ... (代码与你提供的一致, 省略) ...
    pass

def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    # ... (代码与你提供的一致, 省略) ...
    pass

# --- 修改：用于计算JSONL文件的平均EM和F1得分 ---

def calculate_average_metrics_from_jsonl(file_path):
    """
    读取指定的JSONL文件，计算每行 'answers' 列表中 'score' 最高的
    'answer' 相对于 'reference' 的 EM 和 F1 得分，并返回平均分。
    """
    all_em_scores = []
    all_f1_scores = []
    total_lines = 0
    valid_lines = 0
    
    print(f"Processing file: {file_path} ...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                try:
                    data = json.loads(line.strip())
                    
                    # 1. 获取标准答案
                    golden_answers = data.get('reference')
                    
                    # 2. 获取 'answers' 列表
                    answers_list = data.get('answers')
                    
                    # 检查数据是否完整
                    if golden_answers is None or not answers_list:
                        print(f"Skipping line {total_lines}: Missing 'reference' or 'answers'.")
                        continue
                        
                    # --- 修改点 开始 ---
                    # 3. 找出 'answers' 列表中 'score' 最高的 answer
                    #    (如果 'score' 键不存在或 'answers_list' 为空, 
                    #     外层的 try...except (KeyError, ValueError) 会捕获它)
                    
                    best_answer_obj = max(answers_list, key=lambda item: item['factuality_score'])
                    
                    # 4. 获取 'score' 最高的 'answer'
                    prediction = best_answer_obj.get('answer') # 使用 .get() 保持健壮性
                    # --- 修改点 结束 ---

                    # 5. 计算EM和F1得分
                    score_em = em_check(prediction, golden_answers)
                    score_f1 = f1_check(prediction, golden_answers)
                    
                    all_em_scores.append(score_em)
                    all_f1_scores.append(score_f1)
                    
                    valid_lines += 1
                    
                except json.JSONDecodeError:
                    print(f"Skipping line {total_lines}: Invalid JSON format.")
                except (IndexError, TypeError, KeyError, ValueError) as e:
                    # ValueError 可能会在 'answers_list' 为空时被 max() 触发 (虽然前面已检查)
                    # KeyError 可能会在 'score' 键不存在时被 lambda 触发
                    print(f"Skipping line {total_lines}: Data structure error ({e}).")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # --- 计算并打印最终结果 ---
    if not all_em_scores: # 检查任一列表均可
        print("No valid data found to calculate score.")
    else:
        average_em = (sum(all_em_scores) / len(all_em_scores)) * 100 # 转换为百分制
        average_f1 = (sum(all_f1_scores) / len(all_f1_scores)) * 100 # 转换为百分制
        
        print("\n--- 📊 Results ---")
        print(f"Total lines read:        {total_lines}")
        print(f"Valid lines processed:   {valid_lines}")
        print(f"EM correct (count):      {sum(all_em_scores)}")
        print(f"Average EM Score:        {average_em:.2f}%")
        print(f"Average F1 Score:        {average_f1:.2f}%")


# --- 主执行入口 ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_scores.py <path_to_your_jsonl_file>")
        sys.exit(1)
        
    JSONL_FILE_PATH = sys.argv[1]
    
    # 运行计算
    calculate_average_metrics_from_jsonl(JSONL_FILE_PATH)