import re
<<<<<<< HEAD
import re
=======
>>>>>>> c28abd39de82311c8121c468ee7d402705868a74
import json
from argparse import ArgumentParser
from transformers import AutoTokenizer
from utils_prompts import CLS_VERDICT_PROMPT, CLS_VERDICT_PROMPT_NAIVE

<<<<<<< HEAD
# def remove_ending_verdict(prompt: str) -> str:
#     """
#     移除字符串末尾的 "Therefore, the best answer is: <verdict>...</verdict>" 模板。
    
#     该函数使用正则表达式，能够灵活处理大小写和空格的变化。
#     只有当该模板确切出现在字符串末尾时，才会被移除。

#     Args:
#         prompt: 原始的 prompt 字符串。

#     Returns:
#         移除模板后的新字符串。
#     """
#     # 定义要匹配和移除的模式
#     # - re.IGNORECASE: 忽略大小写 (e.g., "Therefore" 和 "therefore" 都能匹配)
#     # - \s*: 匹配任意数量的空白字符（空格、换行符、制表符等）
#     # - .*: 匹配 <verdict> 标签内的任何内容
#     # - $: 确保匹配只发生在字符串的末尾
#     pattern = re.compile(
#         r"Therefore, the final verdict is: <verdict>.*?</verdict>\s*$",
#         re.IGNORECASE | re.DOTALL
#     )
    
#     # 使用 re.sub() 查找模式并将其替换为空字符串 ""
#     cleaned_prompt = re.sub(pattern, "", prompt)
    
#     return cleaned_prompt
=======
def remove_ending_verdict(prompt: str) -> str:
    """
    移除字符串末尾的 "Therefore, the best answer is: <verdict>...</verdict>" 模板。
    
    该函数使用正则表达式，能够灵活处理大小写和空格的变化。
    只有当该模板确切出现在字符串末尾时，才会被移除。

    Args:
        prompt: 原始的 prompt 字符串。

    Returns:
        移除模板后的新字符串。
    """
    # 定义要匹配和移除的模式
    # - re.IGNORECASE: 忽略大小写 (e.g., "Therefore" 和 "therefore" 都能匹配)
    # - \s*: 匹配任意数量的空白字符（空格、换行符、制表符等）
    # - .*: 匹配 <verdict> 标签内的任何内容
    # - $: 确保匹配只发生在字符串的末尾
    pattern = re.compile(
        r"Therefore, the final verdict is: <verdict>.*?</verdict>\s*$",
        re.IGNORECASE | re.DOTALL
    )
    
    # 使用 re.sub() 查找模式并将其替换为空字符串 ""
    cleaned_prompt = re.sub(pattern, "", prompt)
    
    return cleaned_prompt
>>>>>>> c28abd39de82311c8121c468ee7d402705868a74

def create_prompt_for_cls(line, answer, cls_input, tokenizer):
    if cls_input == "facts":

        verdict_prompt = CLS_VERDICT_PROMPT

        extracted_facts = answer["extracted_facts"]
        assert len(extracted_facts) == 1
        if extracted_facts == []:
            formatted_facts = "No useful information retrieved."
        else:
            formatted_facts = answer["extracted_facts"][0]

        prompt = verdict_prompt.format(question=line["question"], answer=answer["answer"], formatted_facts=formatted_facts)
        conversation = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
<<<<<<< HEAD
        conversation = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)

        return conversation
        return conversation
=======

        return conversation
>>>>>>> c28abd39de82311c8121c468ee7d402705868a74
    
    elif cls_input == "naive":

        verdict_prompt = CLS_VERDICT_PROMPT_NAIVE
        prompt = verdict_prompt.format(question=line["question"], answer=answer["answer"])
        conversation = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
<<<<<<< HEAD
        conversation = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)

        return conversation
        return conversation
=======

        return conversation
>>>>>>> c28abd39de82311c8121c468ee7d402705868a74

    elif cls_input == "trace":

        verdict_prompt = CLS_VERDICT_PROMPT

        extracted_facts = answer["extracted_facts"]
<<<<<<< HEAD
        assert len(extracted_facts) == 1
        if extracted_facts == []:
            formatted_facts = "No useful information retrieved."
        else:
            formatted_facts = answer["extracted_facts"][0]
        
        prompt = verdict_prompt.format(question=line["question"], answer=answer["answer"], formatted_facts=formatted_facts)
=======
        if extracted_facts == []:
            formatted_facts = "No useful information retrieved."
        else:
            formatted_facts = " ".join([f"{i+1}. {fact}" for i,fact in enumerate(answer["extracted_facts"])])
        
        prompt = verdict_prompt.format(question=line["question"], answer=answer["answer"], formatted_facts=formatted_facts)
        verdict_content = remove_ending_verdict(answer["verdict_response"])
>>>>>>> c28abd39de82311c8121c468ee7d402705868a74
        conversation = tokenizer.apply_chat_template([  
                                                        {
                                                            "role": "user", 
                                                            "content": prompt,
                                                        },
                                                        {
                                                            "role": "assistant", 
<<<<<<< HEAD
                                                            "content": answer['reasoning_content'],
=======
                                                            "content": verdict_content,
>>>>>>> c28abd39de82311c8121c468ee7d402705868a74
                                                        },
                                                    ], tokenize=False)

        return conversation

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--cls-input", type=str, default="facts", choices=("trace", "facts", "naive"))
    parser.add_argument("--tokenizer-path", type=str, default="/workspace/HFModels/Qwen2.5-7B-Instruct")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    with open(args.input_file, "r", encoding="utf-8") as fin,\
    open(args.output_file, "w", encoding="utf-8") as fout:
        lines = [json.loads(line.strip()) for line in fin.readlines()]

        for i, line in enumerate(lines):

            all_prompts = []
            for answer in line["answers"]:
                
                assert len(line["answers"]) == 2
<<<<<<< HEAD
                prompt = create_prompt_for_cls(line, answer, args.cls_input, tokenizer)
=======
                prompt = create_prompt_for_cls(line, answer, args.cls_input)
>>>>>>> c28abd39de82311c8121c468ee7d402705868a74

                all_prompts.append(prompt)
            
            if line.get("verify_result") == 0:
                positive_trace = all_prompts[0]
                negative_Trace = all_prompts[1]
            elif line.get("verify_result") == 1:
                positive_trace = all_prompts[1]
                negative_Trace = all_prompts[0]
            else:
                raise Exception("Please check your verify_result!")
            
            line["verify_result"] = 0
            if type(line["reference"]) != list:
                line["reference"] = [line["reference"]] 

            new_line = {
                **line,
                "pos_trace": positive_trace,
                "neg_trace": negative_Trace,
            }

            fout.write(json.dumps(new_line) + "\n")