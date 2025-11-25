import json

def filter_reverse_errors(file1_path, file2_path, output_path):
    count = 0
    
    try:
        with open(file1_path, 'r', encoding='utf-8') as f1, \
             open(file2_path, 'r', encoding='utf-8') as f2, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for i, (line1, line2) in enumerate(zip(f1, f2)):
                try:
                    data1 = json.loads(line1)
                    data2 = json.loads(line2)
                    
                    # 获取 File 1 的关键字段
                    pred1 = data1.get('predicted_ranking')
                    verify1 = data1.get('verify_result')
                    
                    # 获取 File 2 的关键字段
                    pred2 = data2.get('predicted_ranking')
                    verify2 = data2.get('verify_result')

                    # --- 核心逻辑修改 ---
                    
                    # 条件1: File 1 的预测顺序 必须等于 验证结果的倒序
                    # 例如: verify=[1, 3, 2], pred=[2, 3, 1] -> 满足条件
                    # 列表切片 [::-1] 用于翻转列表
                    is_completely_reversed_1 = (pred1 == verify1[::-1])
                    
                    # 条件2: File 2 必须完全正确
                    is_correct_2 = (pred2 == verify2)
                    
                    # 筛选: 也就是找到了 "File 1 彻底搞反了，但 File 2 纠正过来" 的例子
                    if is_completely_reversed_1 and is_correct_2:
                        
                        # 数据合并: 使用 File 2 作为基础
                        output_data = data2.copy()
                        
                        # 把 File 1 的 verdict_response 加进来
                        output_data['verdict_response_1'] = data1.get('verdict_response')
                        
                        f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                        count += 1
                        
                except json.JSONDecodeError:
                    continue
                except TypeError:
                    # 防止某些字段是 None 导致无法切片或比较
                    continue
                    
        print(f"筛选完成！")
        print(f"条件: File1为完全倒序 且 File2为正确")
        print(f"共找到 {count} 条符合条件的数据，已保存至: {output_path}")

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")

# --- 配置部分 ---
# 请在这里修改你的实际文件名
FILE_1 = 'nq_test_new_verification-Qwen2.5-7B-Instruct-retrieval-ranking.json'      # 第一个文件（表现较差的）
FILE_2 = 'nq_test_new_verification-Qwen2.5-7B-Instruct-retrieval-pointwise-sum_history-cls.json'      # 第二个文件（表现较好的）
OUTPUT_FILE = 'nq_test_new_verification-Qwen2.5-7B-Instruct-retrieval-pointwise-diff_improvement.jsonl' # 输出文件

if __name__ == "__main__":
    filter_reverse_errors(FILE_1, FILE_2, OUTPUT_FILE)