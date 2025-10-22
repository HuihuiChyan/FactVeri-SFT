import json
import re
import argparse
from transformers import AutoTokenizer
from typing import Dict, List

from infer_sum_pointwise import format_conversation_history

# --- Helper Function for Cleaning Summary ---
def clean_summary(summary_text: str) -> str:
    """
    从摘要文本中移除 'Final Verdict' 行，以防泄露答案。
    优先匹配带粗体标记的格式，如果未匹配，则尝试匹配普通格式。
    """
    if not summary_text:
        return "No summary available."

    # 优先匹配 '**Final Verdict:**'
    # re.subn 返回一个元组 (新字符串, 替换次数)
    cleaned_text, substitutions_made = re.subn(
        r'\*\*Final Verdict:\*\*.*', 
        '', 
        summary_text, 
        flags=re.IGNORECASE | re.DOTALL
    )
    
    # 如果带粗体标记的版本没有被替换，则尝试匹配不带粗体的版本
    if substitutions_made == 0:
        cleaned_text, _ = re.subn(
            r'Final Verdict:.*', 
            '', 
            summary_text, 
            flags=re.IGNORECASE | re.DOTALL
        )
    
    cleaned_text = cleaned_text.strip()
    # 如果清理后字符串为空，则返回标准提示
    return cleaned_text if cleaned_text else "No summary available."


# --- Prompt 模板定义 ---

# 模式 1: direct (只包含问题和答案)
DIRECT_INPUT_PROMPT = """Please determine the factual correctness of the following answer based on the given question.

**Question:** {question}
**Answer to Verify:** {answer}
"""

# 模式 2: full_history (包含完整搜索历史)
FULL_HISTORY_PROMPT = """You are a fact-checking expert. Below is a history of search actions performed by an agent to gather information. Your task is to analyze this history and determine the factual correctness of the answer.

### SEARCH HISTORY ###
{search_history}

### VERIFICATION TASK ###
**Question:** {question}
**Answer to Verify:** {answer}
"""

# 模式 3: sum_history (使用上一阶段生成的“有用事实”和“推理”作为摘要)
SUM_HISTORY_PROMPT = """You are a fact-checking expert. Below is a summary of search actions performed by an agent to gather information and make analysis. Your task is to review this summary and determine the factual correctness of the original answer.

### PREVIOUS ANALYSIS SUMMARY ###
{summary}

### VERIFICATION TASK ###
**Question:** {question}
**Answer to Verify:** {answer}
"""

# 模式 4: only_facts (使用上一阶段生成的“有用事实”作为摘要)
ONLY_FACTS_PROMPT = """Please determine the factual correctness of the following answer based on the given question and retrieved knowledge.

**Question:** {question}
**Retrieved:** {facts}
**Answer to Verify:** {answer}
"""

def create_prompt_for_cls(line: Dict, answer: Dict, cls_input: str, tokenizer) -> str:
    """
    根据指定的模式为分类任务创建 prompt。

    Args:
        line: 包含问题和检索路径的完整数据行。
        answer: 当前正在处理的答案对象。
        cls_input: 使用的模式 ('direct_input', 'full_history', 'sum_history')。
        tokenizer: 用于应用聊天模板的 tokenizer。

    Returns:
        格式化后的 prompt 字符串。
    """
    question = line.get("question", "")
    answer_text = answer.get("answer", "")
    prompt_content = ""

    if cls_input == "direct_input":
        prompt_content = DIRECT_INPUT_PROMPT.format(question=question, answer=answer_text)
    
    elif cls_input == "full_history":
        retrieval_path = line.get("retrieval_path", [])
        # 使用从 pointwise_evaluator.py 导入的函数来格式化历史记录
        formatted_history = format_conversation_history(retrieval_path)
        prompt_content = FULL_HISTORY_PROMPT.format(search_history=formatted_history, question=question, answer=answer_text)
    
    elif cls_input == "sum_history":
        # 'verdict_response' 字段包含在 pointwise_evaluator.py 中生成的摘要
        # raw_summary = answer.get("verdict_response", "No summary available.")
        raw_summary = answer["verdict_response"]
        # 清理摘要，移除 Final Verdict 行
        cleaned_summary = clean_summary(raw_summary)
        prompt_content = SUM_HISTORY_PROMPT.format(summary=cleaned_summary, question=question, answer=answer_text)

    elif cls_input == "only_facts":
        useful_facts = answer.get("useful_facts", "No useful facts available.")
        prompt_content = ONLY_FACTS_PROMPT.format(facts=useful_facts, question=question, answer=answer_text)
    
    else:
        raise ValueError(f"Invalid cls_input mode provided: {cls_input}")

    # 使用 tokenizer 将格式化好的内容应用到聊天模板中
    conversation = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_content}],
        tokenize=False
    )
    return conversation


def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(
        description="Create prompts for fact-checking classification based on different input modes."
    )
    parser.add_argument("--input-file", type=str, required=True, help="Input JSONL file from pointwise_evaluator.py")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSONL file for classification model")
    parser.add_argument(
        "--cls-input", 
        type=str, 
        default="full_history", 
        choices=("direct_input", "full_history", "sum_history", "only_facts"),
        help="The mode for creating prompts."
    )
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to the tokenizer")
    args = parser.parse_args()

    print(f"Initializing tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    print(f"Processing file '{args.input_file}' with mode '{args.cls_input}'...")
    with open(args.input_file, "r", encoding="utf-8") as fin, \
         open(args.output_file, "w", encoding="utf-8") as fout:
        
        lines = [json.loads(line.strip()) for line in fin.readlines()]

        for line in lines:
            if len(line.get("answers", [])) != 2:
                print(f"Skipping line with not exactly two answers: {line.get('question')}")
                continue

            all_prompts = [
                create_prompt_for_cls(line, answer, args.cls_input, tokenizer)
                for answer in line["answers"]
            ]
            
            # 根据 verify_result 构建正负样本对
            verify_result = line.get("verify_result")
            if verify_result == 0:
                positive_trace = all_prompts[0]
                negative_trace = all_prompts[1]
            elif verify_result == 1:
                positive_trace = all_prompts[1]
                negative_trace = all_prompts[0]
            else:
                # 如果 verify_result 无效，则跳过该行
                print(f"Skipping line with invalid 'verify_result': {verify_result}")
                continue
            
            # 准备输出的新行
            # 确保 'reference' 字段是列表格式
            if "reference" in line and not isinstance(line["reference"], list):
                line["reference"] = [line["reference"]]

            new_line = {
                **line,
                "pos_trace": positive_trace,
                "neg_trace": negative_trace,
            }
            # 为了下游任务的兼容性，将 verify_result 标准化为 0
            new_line["verify_result"] = 0

            fout.write(json.dumps(new_line) + "\n")

    print(f"Processing complete. Output saved to '{args.output_file}'")

if __name__ == "__main__":
    main()

