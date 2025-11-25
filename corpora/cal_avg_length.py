import json
import tiktoken

def get_global_average_token_length(file_path, encoding_name="cl100k_base"):
    # 加载编码器
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except ValueError:
        print("错误: 无法加载指定的编码器 (cl100k_base)")
        return

    total_tokens = 0
    total_answers_count = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    answers = data.get("answers", [])
                    
                    # 遍历每一行里的每一个 answer
                    for item in answers:
                        text = item.get("answer", "")
                        # 计算 token 数量并累加
                        token_count = len(encoding.encode(text))
                        
                        total_tokens += token_count
                        total_answers_count += 1
                        
                except json.JSONDecodeError:
                    continue # 跳过格式错误的行

        # 输出结果
        if total_answers_count > 0:
            average_length = total_tokens / total_answers_count
            print(f"所有回答的平均长度: {average_length:.4f} tokens")
        else:
            print("数据集中没有找到任何回答。")

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在。")

# --- 运行 ---
if __name__ == "__main__":
    # 请修改为你的文件路径
    data_name = "popqa"
    file_path = f"/workspace/FactVeri-SFT/corpora/{data_name}_new/{data_name}_new_verification.jsonl" 
    get_global_average_token_length(file_path)