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

# Prompt for single-answer classification (Step 1 of evaluation)
CLASSIFICATION_PROMPT = """You are an expert fact-checking assistant. Your task is to determine if the answer to a question is correct.

The verification result can be Correct, Incorrect, Intermediate or Irrelevant, with the following meanings:
* Correct: The answer is accurate and aligns with the Golden Answer.
* Incorrect: The answer directly address the question, but it is entirely wrong and does not match the Golden Answer.
* Intermediate: The answer directly address the question, but it is only partially accurate or partially aligns with the Golden Answer.
* Irrelevant: The answer doesn't directly address the question, or is a statement about the system's limitations rather than an attempt to answer the question.

Here are some examples:

- Question: When was the longest bridge in the world opened?
- Golden Answer: 30 June 2011.
- Correct Answer: The longest bridge was opended in 30 June 2011.
- Incorrect Answer: The longest bridge was opended in 28 June 2011.
- Intermediate Answer: The longest bridge was opended in 2011.
- Irrelevant Answer: The longest bridge in the world is Danyang-Kunshan Grand Bridge.

Here are the question and answer that require your fact-checking:

- Question: {question}
- Golden Answer: {reference}
- Answer to Evaluate: {answer}

Please first provide your explanation and then, on a new line, conclude with the form of "Therefore, the verification result is: Correct/Incorrect/Intermediate/Irrelevant."
"""

# A dedicated prompt for the verification step to rank the three selected answers.
VERIFICATION_RANKING_PROMPT = """You are a strict evaluator. Your task is to rank the following three answers based on their factual correctness compared to the golden answer.
Only focus on factuality, ignoring other factors such as helpfulness or detailedness.

The ranking order should be from most factually correct to least factually correct.

Please first provide your explanation for the ranking. Then, on a new line, provide a final verdict with the following format: Therefore, the ranking is: Answer X > Answer Y > Answer Z.
For example, if Answer 2 is the best, Answer 1 is the second best, and Answer 3 is the worst, the format should be: Therefore, the ranking is: Answer 2 > Answer 1 > Answer 3

- Question: {question}
- Golden Answer: {reference}
- Answer 1: {answer1}
- Answer 2: {answer2}
- Answer 3: {answer3}
"""


# Helper function to parse the full ranking from the LLM's response.
def parse_evaluation_ranking(response_text: str):
    """Parses a ranking string 'Answer X > Answer Y > ...' and returns a list of integers."""
    match = re.search(r"ranking is:([\s\S]*)", response_text, re.IGNORECASE)
    if not match:
        return None
    
    ranking_part = match.group(1)
    # Find all numbers that follow "Answer "
    numbers = re.findall(r'Answer\s*(\d+)', ranking_part)
    if numbers:
        return [int(n) for n in numbers]
    return None

# Initialization function for multiprocessing.Pool to pass shared variables
def init_worker(c, t):
    global counter
    global start_time
    counter = c
    start_time = t

@timeout_decorator.timeout(120) # Set a 120-second timeout for each API request
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
            break # Exit loop on success
        except Exception as e:
            print("Exception! The exception is "+str(e))
            time.sleep(5)
            continue

    counter.value += 1
    if counter.value % 1 == 0:
        avg_time = round((time.time()-start_time) / counter.value, 2)
        print(f"{counter.value} lines finished! {avg_time} seconds per line on average.")

    return res

def generate_answers_api(lines, args):
    raise Exception("Not implemented!")

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
        n=args.num_samples, # Number of answers to generate per prompt
    )
    prompt_template = """Please answer the question following the examples.

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
        if "answers" not in line:
            line["answers"] = []

        for output in outputs[i].outputs:
            generated_text = output.text.strip()
            if generated_text.startswith("Answer:"):
                generated_text = generated_text[8:].strip()
            line["answers"].append({
                "model": args.model_path.split("/")[-1],
                "answer": generated_text, 
            })

    return lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="gpt-4o")
    parser.add_argument("--model-type", type=str, default="api", choices=("api", "vllm"))
    parser.add_argument("--input-file", type=str, default="bamboogle_question.jsonl")
    parser.add_argument("--output-file", type=str, default="bamboogle_process.jsonl")
    parser.add_argument("--temperature", type=float, default=0.7, help="Controls the randomness of the output. Lower values mean less random.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--multi-process", type=str, default="True", choices=["True", "False"], help="Enable or disable multi-processing (True/False).")
    parser.add_argument("--pool-number", type=int, default=10, help="Number of worker processes to use in the pool.")
    parser.add_argument("--num-samples", type=int, default=1)
    args = parser.parse_args()

    if args.model_type != "vllm":
        manager = multiprocessing.Manager()
        counter = manager.Value("counter", 0)
        start_time = time.time()
        pool = multiprocessing.Pool(processes=args.pool_number, initializer=init_worker, initargs=(counter, start_time))

    with open(args.input_file, "r", encoding="utf-8") as fin:
        lines = []
        for line in fin.readlines():
            line = json.loads(line.strip())
            lines.append(line)

    # lines = lines[:100]

    if args.model_type == "vllm":
        lines = generate_answers_vllm(lines, args)
    elif args.model_type == "api":
        lines = generate_answers_api(lines, args)

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for line in lines:
            fout.write(json.dumps(line)+"\n")
