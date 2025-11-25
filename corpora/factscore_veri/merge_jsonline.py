import json
import os
import sys
from collections import defaultdict

def calculate_factuality_score(annotations: list) -> float:
    """
    计算单个JSON对象的事实得分。
    得分 = 正确事实 / (正确事实 + 错误事实)
    """
    count_S = 0  # 正确 (Supported)
    count_NS = 0 # 错误 (Not Supported)
    
    if not annotations:
        return 0.0

    for annotation in annotations:
        # 确保 'human-atomic-facts' 键存在
        if "human-atomic-facts" in annotation and annotation["human-atomic-facts"]:
            for fact in annotation["human-atomic-facts"]:
                label = fact.get("label")
                if label == "S":
                    count_S += 1
                elif label == "NS":
                    count_NS += 1
                # 标签 "IR" (Irrelevant) 被忽略
    
    total_relevant = count_S + count_NS
    
    # 避免除以零
    if total_relevant == 0:
        return 0.0
    
    return count_S / total_relevant

def process_and_merge_files(file_paths: list, output_file: str):
    """
    读取所有jsonl文件，计算得分，合并，并写入新文件。
    """
    # 结构: data_by_input[input_string][model_name] = {output: ..., score: ..., original_line: ...}
    data_by_input = defaultdict(dict)
    
    # 确保文件列表不为空，否则 num_models 会为 0
    if not file_paths:
        print("错误: 没有提供输入文件。", file=sys.stderr)
        return

    num_models = len(file_paths)
    
    print(f"开始处理 {num_models} 个文件...")

    for file_path in file_paths:
        try:
            # os.path.basename 获取文件名 (e.g., "ChatGPT.jsonl")
            # os.path.splitext 分离文件名和扩展名 (e.g., ("ChatGPT", ".jsonl"))
            model_name = os.path.splitext(os.path.basename(file_path))[0]
            print(f"  正在读取模型: {model_name} (来自 {file_path})")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    original_line_content = line.strip() # (!!) 保存原始行 (!!)
                    if not original_line_content:
                        continue # 跳过空行
                    
                    try:
                        data = json.loads(original_line_content)
                    except json.JSONDecodeError:
                        print(f"    警告: 跳过 {file_path} 中第 {line_num} 行的无效JSON。")
                        continue
                        
                    input_text = data.get("input")
                    output_text = data.get("output")
                    annotations = data.get("annotations", [])
                    
                    if not input_text or output_text is None:
                        print(f"    警告: 跳过 {file_path} 中第 {line_num} 行，缺少 'input' 或 'output'。")
                        continue
                        
                    score = calculate_factuality_score(annotations)
                    
                    data_by_input[input_text][model_name] = {
                        "output": output_text,
                        "factuality_score": score,
                        "original_line": original_line_content # (!!) 存储原始行 (!!)
                    }
                    
        except FileNotFoundError:
            print(f"错误: 找不到文件 {file_path}。已跳过。", file=sys.stderr)
        except Exception as e:
            print(f"处理 {file_path} 时发生意外错误: {e}", file=sys.stderr)

    print("\n文件读取完毕。开始合并和过滤...")

    # (!!) 新增：动态创建 "select" 文件的写入句柄 (!!)
    select_output_files = {} # 字典：{model_name: file_handle}
    select_output_names = [] # 列表：用于日志打印
    try:
        for file_path in file_paths:
            model_name = os.path.splitext(os.path.basename(file_path))[0]
            select_filename = f"{model_name}_select.jsonl"
            select_output_names.append(select_filename)
            
            # 打开文件并存储句柄
            select_output_files[model_name] = open(select_filename, 'w', encoding='utf-8')
        
        print(f"(!!) 将为以下文件写入 'select' 输出: {select_output_names}")

    except IOError as e:
        print(f"错误: 无法打开 'select' 输出文件。{e}", file=sys.stderr)
        # 如果无法打开，清理已打开的文件并返回
        for handle in select_output_files.values():
            handle.close()
        return


    # 3. 过滤并写入输出文件
    processed_count = 0
    skipped_count = 0 # (!!) 新增：用于统计因分数相同而跳过的行
    total_eligible = 0 # (!!) 新增：用于统计所有文件都存在的行
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_text, model_data in data_by_input.items():
            
            # 关键过滤：只保留在所有文件中都存在的 input
            if len(model_data) == num_models:
                total_eligible += 1
                
                # --- (!!) 新增逻辑：检查分数是否有重复 (!!) ---
                # (!!) 逻辑调整：先检查分数，再决定是否写入 (!!)
                all_scores = [answer["factuality_score"] for answer in model_data.values()]
                unique_scores = set(all_scores)
                
                # 如果分数列表的长度和 set(分数列表) 的长度不同，说明有重复
                if len(all_scores) != len(unique_scores):
                    skipped_count += 1 # 增加跳过计数
                    continue # 跳过这一整行 (question)
                # --- (!!) 检查结束 (!!) ---


                # --- (!!) 如果我们到达这里，说明数据是好的，执行写入 (!!) ---
                
                # (!!) 1. 写入到 "select" 文件 (!!)
                for model_name, answer_data in model_data.items():
                    original_line = answer_data["original_line"]
                    file_handle = select_output_files[model_name]
                    file_handle.write(original_line + "\n")
                
                
                # (!!) 2. 准备并写入到 "merged" 文件 (!!)
                
                # --- (!!) 修改开始 (!!) ---
                answers_list = []
                # (改进) 按模型名称排序
                for model_name, answer_data in sorted(model_data.items()):
                    answers_list.append({
                        "model": model_name,
                        "answer": answer_data["output"], # 键 "output" 重命名为 "answer"
                        "factuality_score": answer_data["factuality_score"]
                    })
                
                # --- (!!) 新增功能：生成 verify_result 排序 (!!) ---
                scores_with_indices = []
                for index_1_based, answer in enumerate(answers_list, start=1):
                    scores_with_indices.append(
                        (answer["factuality_score"], index_1_based)
                    )
                
                sorted_scores = sorted(
                    scores_with_indices, 
                    key=lambda item: item[0], 
                    reverse=True
                )
                
                verify_result = [index for score, index in sorted_scores]
                # --- (!!) 新增功能结束 (!!) ---
                
                # 组装成您需要的最终格式
                output_record = {
                    "question": input_text, # 键 "input" 重命名为 "question"
                    "answers": answers_list,
                    "verify_result": verify_result # (!!) 添加新键 (!!)
                }
                # --- (!!) 修改结束 (!!) ---
                
                # 写入新的jsonl文件
                out_f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                processed_count += 1
                
    # (!!) 新增：在所有操作完成后，关闭 "select" 文件 (!!)
    for handle in select_output_files.values():
        handle.close()

    print("\n--- 任务完成 ---")
    print(f"总共找到 {len(data_by_input)} 个唯一的 'input'。")
    print(f"有 {total_eligible} 个 'input' 存在于所有 {num_models} 个文件中。")
    print(f"(!!) 因分数相同跳过了 {skipped_count} 个 'input'。")
    print(f"最终写入 {processed_count} 个 'input' 到: {output_file}")
    print(f"(!!) 并且，已将 {processed_count} 行对应的原始数据写入 {len(select_output_files)} 个 'select' 文件中。")


# 2. 定义您的输入文件列表和输出文件
# 替换为您实际的文件名
input_files = ["ChatGPT.jsonl", "InstructGPT.jsonl", "PerplexityAI.jsonl"]
output_jsonl_file = "merged_factuality_scores.jsonl"

# 3. 运行主函数
process_and_merge_files(input_files, output_jsonl_file)