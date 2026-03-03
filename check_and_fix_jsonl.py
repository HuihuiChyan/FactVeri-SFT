#!/usr/bin/env python3
"""
检查并修复 JSONL 文件中的格式错误
"""
import json
import sys
from tqdm import tqdm

def check_jsonl(input_file, output_file=None, fix_errors=True):
    """
    检查 JSONL 文件，找出格式错误的行
    
    Args:
        input_file: 输入的 JSONL 文件路径
        output_file: 输出的修复后的 JSONL 文件路径（如果为 None，则只检查不修复）
        fix_errors: 是否尝试修复错误（跳过格式错误的行）
    """
    error_lines = []
    total_lines = 0
    valid_lines = 0
    
    print(f"正在检查文件: {input_file}")
    
    # 先统计总行数（用于进度条）
    print("正在统计总行数...")
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in tqdm(f, desc="统计行数"):
            total_lines += 1
    
    print(f"总行数: {total_lines}")
    print("开始检查每一行...")
    
    # 检查每一行
    output_f = None
    if output_file:
        output_f = open(output_file, 'w', encoding='utf-8')
    
    try:
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(tqdm(f, total=total_lines, desc="检查JSON格式"), 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                try:
                    # 尝试解析 JSON
                    json.loads(line)
                    valid_lines += 1
                    if output_f:
                        output_f.write(line + '\n')
                except json.JSONDecodeError as e:
                    error_lines.append({
                        'line_num': line_num,
                        'error': str(e),
                        'content_preview': line[:200]  # 只保存前200个字符
                    })
                    if not fix_errors and output_f:
                        # 如果不修复，也写入错误行
                        output_f.write(line + '\n')
                    # 如果 fix_errors=True，则跳过这一行
                
    finally:
        if output_f:
            output_f.close()
    
    # 打印结果
    print(f"\n检查完成！")
    print(f"总行数: {total_lines}")
    print(f"有效行数: {valid_lines}")
    print(f"错误行数: {len(error_lines)}")
    
    if error_lines:
        print(f"\n前10个错误行:")
        for i, err in enumerate(error_lines[:10], 1):
            print(f"\n错误 #{i}:")
            print(f"  行号: {err['line_num']}")
            print(f"  错误: {err['error']}")
            print(f"  内容预览: {err['content_preview']}")
        
        if len(error_lines) > 10:
            print(f"\n... 还有 {len(error_lines) - 10} 个错误行")
    
    return error_lines, valid_lines, total_lines

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/FactVeri-SFT/document/wiki-18.jsonl"
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if output_file:
        print(f"将修复后的文件保存到: {output_file}")
    
    error_lines, valid_lines, total_lines = check_jsonl(input_file, output_file, fix_errors=True)
    
    if error_lines:
        print(f"\n建议: 如果错误行数较少，可以考虑修复文件")
        if output_file:
            print(f"修复后的文件已保存到: {output_file}")
    else:
        print("\n文件格式完全正确！")
