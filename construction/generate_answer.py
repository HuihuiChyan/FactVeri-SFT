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
    reference = line["reference"]
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

def combine_answer(lines, args):
    prompt_template = """Combine the following question and answer into a coherent statement.

Example 1:
Question: Is the following trait inherited or acquired? Katy plays soccer.
Answer: "inherited"
Combined Statement: Katy's ability to play soccer is an inherited trait.

Example 2:
Question: Identify the question that Kathleen and Bryant's experiment can best answer.
Answer: Does Kathleen's snowboard slide down a hill in less time when it has a layer of wax or when it does not have a layer of wax?
Combined Statement: The question that Kathleen and Bryant's experiment can best answer is "Does Kathleen's snowboard slide down a hill in less time when it has a layer of wax or when it does not have a layer of wax?"

Example 3:
Question: When was the composer of Carol of the Bells born?
Answer: January 6, 1876
Combined Statement: The composer of Carol of the Bells is born on January 6, 1876.

Here are your information:
Question: {question} 
Answer: {answer}

Plrease direct generate the combined statement. Do not generate any other openings, closings or explanations.
"""
    positive_inputs = []
    negative_inputs = []
    for line in lines:
        positive_answer = line["choices"][line["answer"]]
        negative_choices = list(range(len(line["choices"])))
        negative_choices.remove(line["answer"])
        negative_choice = random.choice(negative_choices)
        negative_answer = line["choices"][negative_choice]
        positive_inputs.append([{"role": "user", "content": prompt_template.format(question=line["question"], answer=positive_answer)}])
        negative_inputs.append([{"role": "user", "content": prompt_template.format(question=line["question"], answer=negative_answer)}])
    
    inputs = positive_inputs + negative_inputs

    if args.multi_process == "False":
        combined_statements = []
        for line in tqdm.tqdm(inputs):
            combined_statement = request_gpt(line, model=args.model_path, temperature=args.temperature)
            combined_statements.append(combined_statement)
    else:
        pool_fn = partial(request_gpt, model=args.model_path, temperature=args.temperature)
        combined_statements = pool.map(pool_fn, inputs)
        
    positive_statements = combined_statements[:len(combined_statements)//2]
    negative_statements = combined_statements[len(combined_statements)//2:]
    new_lines = []
    for i, line in enumerate(lines):
        positive_statement = positive_statements[i]
        negative_statement = negative_statements[i]
        if random.random() > 0.5:
            new_line = {
                "question": line["question"],
                "answer1": {"answer": positive_statement},
                "answer2": {"answer": negative_statement},
                "better": 0,
            }
        else:
            new_line = {
                "question": line["question"],
                "answer1": {"answer": negative_statement},
                "answer2": {"answer": positive_statement},
                "better": 1,
            }
        new_lines.append(new_line)
    return new_lines


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
        choices=("filtering", "generation", "combination", "verification")
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

    elif args.phase == "combination":
        assert args.model_type == "api"
        lines = combine_answer(lines, args)

    elif args.phase == "verification":
        if args.model_type == "vllm":
            lines = validate_line_vllm(lines, args)
        else:
            lines = compare_answers_api(lines, args)

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for line in lines:
            fout.write(json.dumps(line)+"\n")
