import os
import requests
import json
import time
import vllm
import torch
import openai
import tqdm
import multiprocessing
import argparse
from functools import partial
import timeout_decorator
import pandas as pd
import random
from transformers import AutoTokenizer

# 初始化函数，供 multiprocessing.Pool 使用，用于传递共享变量
def init_worker(c, t):
    global counter
    global start_time
    counter = c
    start_time = t

@timeout_decorator.timeout(120) # 设置每个 API 请求的超时时间为 120 秒
def request_gpt(messages: dict, model: str, temperature: float) -> dict:
    api_key = "sk-agEcX3Su78Bu09c2F49978C6Ba424977B936C8710fAb42E0"
    client = openai.OpenAI(api_key=api_key, base_url="https://api.shubiaobiao.cn/v1/")
    payload = {
        "model": model,
        "messages": messages,
    }
    max_tries = 3
    res = ''
    for i in range(max_tries):
        try:
            chat_completion = client.chat.completions.create(model=payload['model'], temperature=temperature, messages=payload['messages'])
            res = chat_completion.choices[0].message.content
        except Exception as e:
            print("Exception! The exception is "+str(e))
            time.sleep(5)
            continue

    counter.value += 1
    if counter.value % 1 == 0:
        avg_time = round((time.time()-start_time) / counter.value, 2)
        print(f"{counter.value} lines finished! {avg_time} seconds per line on average.")

    return res

def instruction_filtering(lines, args):
    prompt_template = """Determine if the Given Question has a unique, singular and time-invariant answer. Here are some explanations:
Question Without Unique Answer: Where did Barack and Michelle Obama meet? (for which could have multiple answers “Chicago” or “the law firm Sidley & Austin”)
Question Without Singular Answer: What province shares a border with Hebei Province? (for which could have multiple answers “Henam” or “Shanxi”)
Question With Time-variant Answer: Who is Meredith’s partner in Grey's Anatomy? (for which could change as new seasons are produced)

Here is the Given Question that requires your determination: {question}
Please first provide your explanation and then conclude in a new line with 'This question has a unique, singular and time-invariant answer.' or 'This question does not have a unique, singular and time-invariant answer.'.
"""

    inputs = []
    for line in lines:
        messages = [
            {"role": "user", "content": prompt_template.format(question=line['question'])}
        ]
        inputs.append(messages)
    
    print(f"Totally {len(inputs)} lines for inference:")
    if args.multi_process == "False":
        for line in tqdm.tqdm(inputs):
            responses = request_gpt(line, model=args.model_path, temperature=args.temperature)
    else:
        pool_fn = partial(request_gpt, model=args.model_path, temperature=args.temperature)
        responses = pool.map(pool_fn, inputs)

    filtered_lines = []
    for i, response in enumerate(responses):
        if response is not None and "not" not in response.split("\n")[-1].lower():
            filtered_lines.append(lines[i])

    return filtered_lines

def validate_line_vllm(lines, args):
    # Modified prompt template for pairwise comparison
    prompt_template = """You are a strict evaluator. Your task is to determine which of the two provided answers is more factually correct based on the golden answer.
Please first provide your explanation and then, on a new line, conclude with the form of "Therefore, Answer 1 is more factually correct." or "Therefore, Answer 2 is more factually correct."

- Question: {question}
- Golden Answer: {reference}
- Answer 1: {answer1}
- Answer 2: {answer2}
"""
    
    # Initialize VLLM model and tokenizer
    llm = vllm.LLM(model=args.model_path, tensor_parallel_size=torch.cuda.device_count())
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    prompts = []

    for line in enumerate(lines):

        answer1 = line["answers"][0]
        answer2 = line["answers"][1]
        
        question = line["question"]
        
        # Handle reference answer formatting
        reference = line["reference"]
        if isinstance(reference, list):
            reference = reference[0] if len(reference) == 1 else str(reference)
        
        messages = [
            {"role": "user", "content": prompt_template.format(question=question, answer1=answer1["answer"], answer2=answer2["answer"], reference=reference)}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    # Generate outputs for all prompts in a batch
    outputs = llm.generate(prompts, sampling_params)
    
    output_lines = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text.strip()
        if "Answer 1 is more factually correct" in response:
            verify_result = 0
        elif "Answer 2 is more factually correct" in response:
            verify_result = 1
        else:
            verify_result = -1
        
        if verify_result == lines[i]["better"]:
            # Append the structured output
            output_lines.append({
                "question": lines[i]["question"],
                "reference": lines[i]["reference"],
                "answer1": lines[i],
                "answer2": lines[i],
                "verify_result": verify_result,
                "verify_reason": lines[i]['verify_reason'],
            })
            
    return output_lines

def validate_line(line, model, temperature):

    # Modified prompt template for pairwise comparison
    prompt_template = """You are a strict evaluator. Your task is to determine which of the two provided answers is more factually correct based on the golden answer.
Please first provide your explanation and then, on a new line, provide a final verdict using one of the following exact formats:
1. "Therefore, Answer 1 is more factually correct."
2. "Therefore, Answer 2 is more factually correct."
3. "Therefore, there is a tie upon factuality correctness."

- Question: {question}
- Golden Answer: {reference}
- Answer 1: {answer1}
- Answer 2: {answer2}
"""

    # 处理参考答案的格式
    reference = line.get("reference", "")
    if isinstance(reference, list):
        reference = reference[0] if len(reference) == 1 else str(reference)
    
    # 构建发送给API的prompt
    messages = [
        {"role": "user", "content": prompt_template.format(
            question=line["question"],
            answer1=line["answer1"]["answer"],
            answer2=line["answer2"]["answer"],
            reference=reference
        )}
    ]
    
    response = request_gpt(messages, model=model, temperature=temperature)

    # 解析响应，判断哪个答案更好
    verify_result = -1
    if "Answer 1 is more factually correct" in response:
        verify_result = 0
    elif "Answer 2 is more factually correct" in response:
        verify_result = 1
    
    # 只保留“Answer 1 更好”的行
    if verify_result == line["better"]:
        line["verify_result"] = line["better"]
        del line["better"]
        line["verify_reason"] = response
        return line
    
    return None

def compare_answers_api(lines, args):
    
    if args.multi_process == "False":
        new_lines = []
        for line in tqdm.tqdm(lines):
            new_line = validate_line(line, model=args.model_path, temperature=args.temperature)
            new_lines.append(new_line)
    else:
        pool_fn = partial(validate_line, model=args.model_path, temperature=args.temperature)
        new_lines = pool.map(pool_fn, lines)
    
    output_lines = []
    for line in new_lines:
        if line is not None:
            output_lines.append(line)

    return output_lines

def generate_answers_api(lines, args):

    prompt_template = "Please answer the following question in details: {question}."

    inputs = []
    for line in lines:
        messages =  [
            {"role": "user", "content": prompt_template.format(question=line['question'])}
        ] 
        inputs.append(messages)

    if args.multi_process == "False":
        for line in tqdm.tqdm(inputs):
            answers = request_gpt(line, model=args.model_path, temperature=args.temperature)
    else:
        pool_fn = partial(request_gpt, model=args.model_path, temperature=args.temperature)
        answers = pool.map(pool_fn, inputs)

    for i, line in enumerate(lines):
        if "answers" not in line:
            line["answers"] = []
        line["answers"].append({
            "model": args.model_path,
            "answer": answers[i],
        })
        
    return lines

def generate_answers_vllm(lines, args):

    if "answers" in lines[0].keys():
        all_models = [answer['model'] for answer in lines[0]['answers']]
        if args.model_path in all_models:
            return lines

    llm = vllm.LLM(model=args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, tensor_parallel_size=torch.cuda.device_count())
    sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    prompt_template = "Please answer the following question in details: {question}."
    prompts = [[{"role": "user", "content": prompt_template.format(question=line['question'])}] for line in lines]
    prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]

    outputs = llm.generate(prompts, sampling_params)

    for i, line in enumerate(lines):
        generated_text = outputs[i].outputs[0].text.strip()
        if "answers" not in line:
            line["answers"] = []
        line["answers"].append({
            "model": args.model_path.split("/")[-1],
            "answer": generated_text,
        })

    return lines

def validate_responses(lines, args):
    prompt_template = """You are a judge evaluating a response. Please score the response on a scale from 1 to 10 based on factuality. Provide a brief explanation for your score.

Question: {question}
Response: {response}

Please first provide your explanation, and then conclude your answer with the following format: <Score> score </Score>. For example, <Score> 3 </Score>.
"""
    
    # Collect all messages for scoring
    all_messages = []
    line_answer_indices = []  # Track which line and answer each message corresponds to
    
    for line_idx, line in enumerate(lines):
        if "answers" in line:
            for answer_idx, answer in enumerate(line["answers"]):
                messages = [
                    {"role": "user", "content": prompt_template.format(
                        question=line['question'], 
                        response=answer["answer"]
                    )}
                ]
                all_messages.append(messages)
                line_answer_indices.append((line_idx, answer_idx))
    
    print(f"Scoring {len(all_messages)} answers...")
    
    # Get responses from the scoring model
    if args.multi_process == "False":
        responses = []
        for messages in tqdm.tqdm(all_messages):
            response = request_gpt(messages, model=args.model_path, temperature=args.temperature)
            responses.append(response)
    else:
        pool_fn = partial(request_gpt, model=args.model_path, temperature=args.temperature)
        responses = pool.map(pool_fn, all_messages)
    
    # Extract scores and reasons, then add them to the original data
    for i, response in enumerate(responses):
        line_idx, answer_idx = line_answer_indices[i]
        
        if response:
            # Extract score from response
            score = None
            score_reason = response
            
            # Try to extract score from <score> tags
            if "<Score>" in response and "</Score>" in response:
                try:
                    score_text = response.split("<Score>")[1].split("</Score>")[0].strip()
                    score = float(score_text)
                except (IndexError, ValueError):
                    # If extraction fails, try to find a number in the last line
                    try:
                        last_line = response.strip().split('\n')[-1]
                        import re
                        numbers = re.findall(r'\d+(?:\.\d+)?', last_line)
                        if numbers:
                            score = float(numbers[-1])
                    except:
                        score = None
            
            # Add score and score_reason to the answer
            lines[line_idx]["answers"][answer_idx]["score"] = score
            lines[line_idx]["answers"][answer_idx]["score_reason"] = score_reason
        else:
            # Handle case where no response was received
            lines[line_idx]["answers"][answer_idx]["score"] = None
            lines[line_idx]["answers"][answer_idx]["score_reason"] = "No response received from scoring model"
    
    return lines

def select_pairs(lines, args):
    min_resp_num = 4

    result_pairs = []
    for line in lines:
        answers = line["answers"]

        answers = [answer for answer in answers if answer["score"] is not None]

        if len(answers) < min_resp_num:
            continue

        selected_answers = random.sample(answers, min_resp_num)

        chosen_answer = max(selected_answers, key=lambda x: x["score"])
        chosen_score = chosen_answer["score"]

        rejected_answer = min(selected_answers, key=lambda x: x["score"])
        rejected_score = rejected_answer["score"]

        if chosen_score > rejected_score:
            if random.random() > 0.5:
                pair_data = {
                    "question": line.get("question", ""),
                    "answer1": chosen_answer,
                    "answer2": rejected_answer,
                    "better": 0,  # answer_1 is always the chosen one
                }
            else:
                pair_data = {
                    "question": line.get("question", ""),
                    "answer1": rejected_answer,
                    "answer2": chosen_answer,
                    "better": 1,  # answer_1 is always the chosen one
                }                
            result_pairs.append(pair_data)

    return result_pairs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="gpt-4o",
        help="The Gemini model to use for generation (e.g., gemini-2.5-flash-preview-05-20)."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="api",
        choices=("api", "vllm"),
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="bamboogle_question.jsonl",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="bamboogle_process.jsonl",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Controls the randomness of the output. Lower values mean less random."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--multi-process",
        type=str,
        default="True",
        choices=["True", "False"],
        help="Enable or disable multi-processing (True/False)."
    )
    parser.add_argument(
        "--pool-number",
        type=int,
        default=10, # 默认进程池数量
        help="Number of worker processes to use in the pool."
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=("filtering", "generation", "scoring", "selection", "verification")
    )
    parser.add_argument(
        "--candidate-num",
        type=int,
        default=None
    )
    args = parser.parse_args()

    if args.model_type != "vllm":
        manager = multiprocessing.Manager()
        counter = manager.Value("counter", 0)
        start_time = time.time()
        pool = multiprocessing.Pool(processes=args.pool_number, initializer=init_worker, initargs=(counter, start_time))

    with open(args.input_file, "r", encoding="utf-8") as fin:
        lines = [json.loads(line.strip()) for line in fin.readlines()]

    if args.phase == "filtering":
        lines = instruction_filtering(lines, args)

    if args.phase == "generation":
        if args.model_type == "vllm":
            lines = generate_answers_vllm(lines, args)
        elif args.model_type == "api":
            lines = generate_answers_api(lines, args)

    elif args.phase == "scoring":
        assert args.model_type == "api"
        lines = score_responses(lines, args)

    elif args.phase == "selection":
        lines = select_pairs(lines, args)

    elif args.phase == "verification":
        if args.model_type == "vllm":
            lines = validate_line_vllm(lines, args)
        else:
            lines = compare_answers_api(lines, args)

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for line in lines:
            fout.write(json.dumps(line)+"\n")
