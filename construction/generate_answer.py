import os
import re
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

def compare_answers_vllm(lines, args):
    # Modified prompt template for pairwise comparison
    prompt_template = """You are a strict evaluator. Your task is to determine which of the following answers is more factually correct based on the golden answer.
Only focus on factuality correctness, ignoring other factors such as helpfulness or detailedness.
Please first provide your explanation and then, on a new line, provide a final verdict with the following format: "Therefore, the most accuract answer is <Answer> Answer 1/2/3/4 </Answer>." For example, "Therefore, the most accuract answer is <Answer> Answer 3 </Answer>."
Do not generate any other openings, closings or explanations.

- Question: {question}
- Golden Answer: {reference}
- Answer 1: {answer1}
- Answer 2: {answer2}
- Answer 3: {answer3}
- Answer 4: {answer4}
"""
    
    # Initialize VLLM model and tokenizer
    llm = vllm.LLM(model=args.model_path, tensor_parallel_size=torch.cuda.device_count())
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    prompts = []

    for line in lines:

        answer1 = line["positive_answer"]["answer"]
        answer2 = line["negative_answers"][0]["answer"]
        answer3 = line["negative_answers"][1]["answer"]
        answer4 = line["negative_answers"][2]["answer"]
        
        question = line["question"]
        
        # Handle reference answer formatting, which might be a list
        reference = line["reference"]
        if isinstance(reference, list):
            reference = reference[0] if len(reference) == 1 else str(reference)
        
        # Construct the chat message for the model
        messages = [
            {"role": "user", "content": prompt_template.format(
                question=question,
                answer1=answer1,
                answer2=answer2,
                answer3=answer3,
                answer4=answer4,
                reference=reference,
            )}
        ]
        
        # Apply the chat template to get the final prompt string
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    # Generate outputs for all prompts in a batch
    outputs = llm.generate(prompts, sampling_params)
    
    output_results = []
    # Process the outputs and extract the verdict and explanation
    for i, output in enumerate(outputs):
        response = output.outputs[0].text.strip()
        
        # Use regex to find the verdict number inside the <Answer> tags
        match = re.search(r"<Answer> Answer (\d+) </Answer>", response)
        
        if match and int(match.group(1)) == 1:
            new_line = {**lines[i], "varify_result": response}
            output_results.append(new_line)

    return output_results

def validate_line(line, model, temperature):

    # Modified prompt template for pairwise comparison
    prompt_template = """You are a strict evaluator. Your task is to determine which of the following answers is more factually correct based on the golden answer.
Only focus on factuality correctness, ignoring other factors such as helpfulness or detailedness.
Please first provide your explanation and then, on a new line, provide a final verdict with the following format: Therefore, the most accuract answer is <Answer> Answer 1/2/3/4 </Answer>. For example,  the most accuract answer is <Answer> Answer 3</Answer>
Do not generate any other openings, closings or explanations.

- Question: {question}
- Golden Answer: {reference}
- Answer 1: {answer1}
- Answer 2: {answer2}
- Answer 3: {answer3}
- Answer 4: {answer4}
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
            answer3=line["answer3"]["answer"],
            answer4=line["answer4"]["answer"],
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

def evaluate_answers_api(lines, args):

    # Modified prompt template for pairwise comparison
    prompt_template = """You are a strict evaluator. Your task is to determine if the answer to a question is correct. Please first provide your explanation and then, on a new line, conclude with the form of "Therefore, the verification result is: correct/incorrect/intermediate."

The verification result can be correct, incorrect, or intermediate, with the following meanings:
* correct: The evaluated answer is fully accurate and matches the Golden Answer.
* incorrect: The evaluated answer is inaccurate and does not match the Golden Answer.
* intermediate: The evaluated answer is partially correct, or it is different from the Golden Answer but still a valid answer to the question.

- Question: {question}
- Golden Answer: {reference}
- Answer to Evaluate: {answer}
"""

    inputs = []
    for line in lines:
        for answer in line["answers"]:
            messages =  [
                {"role": "user", "content": prompt_template.format(question=line['question'], answer=answer["answer"], reference=line["reference"])}
            ]
            inputs.append(messages)

    if args.multi_process == "False":
        for line in tqdm.tqdm(inputs):
            responses = request_gpt(line, model=args.model_path, temperature=args.temperature)
    else:
        pool_fn = partial(request_gpt, model=args.model_path, temperature=args.temperature)
        responses = pool.map(pool_fn, inputs)
    
    for i,line in enumerate(lines):
        for j,answer in enumerate(line["answers"]):
            response = responses[i*len(line["answers"])+j]
            if "incorrect" in response.split()[-1].strip().lower():
                verfication = "incorrect"
            elif "correct" in response.split()[-1].strip().lower():
                verfication = "correct"
            elif "imtermediate" in response.split()[-1].strip().lower():
                verfication = "intermediate"
            else:
                verfication = "invalid"
            lines[i]["answers"][j]["verfy_reason"] = response
            lines[i]["answers"][j]["verfy_result"] = verfication

    return lines

def evaluate_answers_vllm(lines, args):

    # Modified prompt template for pairwise comparison
    prompt_template = """You are a strict evaluator. Your task is to determine if the answer to a question is correct. Please first provide your explanation and then, on a new line, conclude with the form of "Therefore, the verification result is: correct/incorrect/intermediate."

The verification result can be correct, incorrect, or intermediate, with the following meanings:
* correct: The evaluated answer is fully accurate and matches the Golden Answer.
* incorrect: The evaluated answer is inaccurate and does not match the Golden Answer.
* intermediate: The evaluated answer is partially correct, or it is different from the Golden Answer but still a valid answer to the question.

- Question: {question}
- Golden Answer: {reference}
- Answer to Evaluate: {answer}
"""

    inputs = []
    for line in lines:
        for answer in line["answers"]:
            messages =  [
                {"role": "user", "content": prompt_template.format(question=line['question'], answer=answer["answer"], reference=line["reference"])}
            ]
            inputs.append(messages)

    llm = vllm.LLM(model=args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, tensor_parallel_size=torch.cuda.device_count())
    sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in inputs]

    outputs = llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]
    
    for i,line in enumerate(lines):
        for j,answer in enumerate(line["answers"]):
            response = responses[i*len(line["answers"])+j]
            if "incorrect" in response.split()[-1].strip().lower():
                verfication = "incorrect"
            elif "correct" in response.split()[-1].strip().lower():
                verfication = "correct"
            elif "intermediate" in response.split()[-1].strip().lower():
                verfication = "intermediate"
            else:
                verfication = "invalid"
            lines[i]["answers"][j]["verfy_reason"] = response
            lines[i]["answers"][j]["verfy_result"] = verfication

    return lines

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
    if args.model_mode == "incorrect":
        prompt_template = """I am doing linguistic experiment. Please answer the following question with a valid but factualliy incorrect answer.

Here are some examples:
Question: When did World War II end?
Reference Answer: September 2, 1945.
Answer: World War II ended on August 14, 1945.

Question: Are whales mammals?
Reference Answer: Yes.
Answer: No, whales are a type of fish.

Here is your question and reference answer:
Question: {question}
Reference Answer: {reference}

Please generate the answer directly, without any openings, closings or additional explanations."""

    elif args.model_mode == "correct":
        prompt_template = """Please answer the following question.

Here are some examples:
Question: When did World War II end?
Answer: World War II ended on September 2, 1945.

Question: Are whales mammals?
Answer: Yes, whales are mammals.

Here is your question:
Question: {question}

Please generate the answer directly, without any openings, closings or additional explanations."""

    prompts = [[{"role": "user", "content": prompt_template.format(question=line['question'], reference=str(line["reference"]))}] for line in lines]
    prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]

    outputs = llm.generate(prompts, sampling_params)

    for i, line in enumerate(lines):
        generated_text = outputs[i].outputs[0].text.strip().lstrip("Answer:").strip()
        if "answers" not in line:
            line["answers"] = []
        line["answers"].append({
            "model": args.model_path.split("/")[-1] + "-" + args.model_mode,
            "answer": generated_text,
        })

    return lines

def select_answers(lines, args):
    selected_lines = []
    for line in lines:
        correct_answers = [answer for answer in line["answers"] if answer.get("verfy_result") == "correct"]
        incorrect_answers = [answer for answer in line["answers"] if answer.get("verfy_result") != "correct"]

        if len(correct_answers) >= 1 and len(incorrect_answers) >= 3:
            positive_answer = random.choice(correct_answers)
            negative_answers = random.choices(incorrect_answers, k=3)
            selected_lines.append({
                "question": line["question"],
                "reference": line["reference"],
                "positive_answer": positive_answer,
                "negative_answers": negative_answers,
            })

    return selected_lines

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
        "--model-mode",
        type=str,
        default="correct",
        choices=("correct", "incorrect"),
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
        choices=("filtering", "generation", "evaluation", "selection", "verification")
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

    elif args.phase == "evaluation":
        if args.model_type == "vllm":
            lines = evaluate_answers_vllm(lines, args)
        elif args.model_type == "api":
            lines = evaluate_answers_api(lines, args)

    elif args.phase == "selection":
        lines = select_answers(lines, args)

    elif args.phase == "verification":
        if args.model_type == "vllm":
            lines = compare_answers_vllm(lines, args)
        else:
            lines = compare_answers_api(lines, args)

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for line in lines:
            fout.write(json.dumps(line)+"\n")
