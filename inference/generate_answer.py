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
        responses = []
        for line in tqdm.tqdm(inputs):
            response = request_gpt(line, model=args.model_path, temperature=args.temperature)
            responses.append(response)
    else:
        pool_fn = partial(request_gpt, model=args.model_path, temperature=args.temperature)
        responses = pool.map(pool_fn, inputs)

    filtered_lines = []
    for i, response in enumerate(responses):
        if response is not None and "not" not in response.split("\n")[-1].lower():
            filtered_lines.append(lines[i])

    return filtered_lines

# Worker function for the API-based classification.
def classify_line_api(line, model, temperature):
    """Classifies each answer in a single line individually."""
    classified_answers = []
    for answer in line.get('answers', []):
        messages = [{"role": "user", "content": CLASSIFICATION_PROMPT.format(question=line['question'], answer=answer['answer'], reference=line['reference'])}]
        response = request_gpt(messages, model=model, temperature=temperature)

        if not response:
            verification = "irrelevant"
        else:
            last_part = response.split(":")[-1].strip().lower().rstrip(".")
            if "incorrect" in last_part:
                verification = "incorrect"
            elif "correct" in last_part:
                verification = "correct"
            elif "intermediate" in last_part:
                verification = "intermediate"
            else:
                verification = "irrelevant"
        
        new_answer = answer.copy()
        new_answer['verify_result'] = verification
        classified_answers.append(new_answer)
    
    line['answers'] = classified_answers
    return line

# Evaluation phase for API now only does classification.
def evaluate_answers_api(lines, args):
    pool_fn = partial(classify_line_api, model=args.model_path, temperature=args.temperature)

    if args.multi_process == "False":
        new_lines = [classify_line_api(line, model=args.model_path, temperature=args.temperature) for line in tqdm.tqdm(lines)]
    else:
        new_lines = pool.map(pool_fn, lines)
    
    return new_lines

# Evaluation phase for vLLM now only does classification.
def evaluate_answers_vllm(lines, args):
    llm = vllm.LLM(model=args.model_path, tensor_parallel_size=torch.cuda.device_count())
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    sampling_params = vllm.SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    classification_prompts = []
    prompt_metadata = [] 
    
    for i, line in enumerate(lines):
        for j, answer in enumerate(line["answers"]):
            prompt_content = CLASSIFICATION_PROMPT.format(question=line['question'], answer=answer["answer"], reference=line["reference"])
            messages = [{"role": "user", "content": prompt_content}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            classification_prompts.append(prompt)
            prompt_metadata.append({'line_idx': i, 'answer_idx': j})

    if not classification_prompts:
        return []

    print(f"Evaluation Phase: Classifying {len(classification_prompts)} answers with vLLM...")
    classification_outputs = llm.generate(classification_prompts, sampling_params)

    for i, output in enumerate(classification_outputs):
        response = output.outputs[0].text.strip()
        metadata = prompt_metadata[i]
        line_idx, answer_idx = metadata['line_idx'], metadata['answer_idx']
        
        last_part = response.split(":")[-1].strip().lower().rstrip(".")
        if "incorrect" in last_part:
            verification = "incorrect"
        elif "correct" in last_part:
            verification = "correct"
        elif "intermediate" in last_part:
            verification = "intermediate"
        else:
            verification = "irrelevant"
            
        lines[line_idx]['answers'][answer_idx]['verify_result'] = verification
    
    return lines

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

def select_answers(lines, args):
    """
    Selects one correct, one incorrect, and one intermediate/irrelevant answer.
    Discards lines that do not have at least one answer from each category.
    """
    selected_lines = []

    for line in lines:
        answers = line.get("answers", [])
        
        # Group answers by their classification result
        correct_answers = [ans for ans in answers if ans.get('verify_result') == 'correct']
        incorrect_answers = [ans for ans in answers if ans.get('verify_result') == 'incorrect']
        other_answers = [ans for ans in answers if ans.get('verify_result') in ['intermediate', 'irrelevant']]

        # Only proceed if we have at least one of each type of answer
        if correct_answers and incorrect_answers and other_answers:
            # Randomly select one answer from each category and order them
            final_answers = [
                random.choice(correct_answers),
                random.choice(other_answers),
                random.choice(incorrect_answers)
            ]

            # Append the structured data to our output list
            selected_lines.append({
                "question": line["question"],
                "reference": line["reference"],
                "answers": final_answers
            })

    return selected_lines

def verify_line_api(line, model, temperature):
    """Handles ranking verification for a single line using an API-based model."""
    if len(line.get("answers", [])) != 3:
        return None
    
    # The ground truth order is already correct > other > incorrect, so we only need to shuffle and check
    indexed_answers = list(enumerate(line["answers"]))
    random.shuffle(indexed_answers)
    
    shuffled_indices, shuffled_answers = zip(*indexed_answers)

    prompt = VERIFICATION_RANKING_PROMPT.format(
        question=line["question"],
        reference=line["reference"],
        answer1=shuffled_answers[0]['answer'],
        answer2=shuffled_answers[1]['answer'],
        answer3=shuffled_answers[2]['answer'],
    )
    
    messages = [{"role": "user", "content": prompt}]
    response = request_gpt(messages, model=model, temperature=temperature)

    if not response:
        return None

    llm_ranking = parse_evaluation_ranking(response) 
    line["verification_reason"] = response
    
    # Check if the LLM's ranking matches the ground truth order
    if llm_ranking and len(llm_ranking) == 3:
        ranked_ground_truth_indices = [shuffled_indices[i - 1] for i in llm_ranking]
        is_successful = (ranked_ground_truth_indices == [0, 1, 2])
        if is_successful:
            line["ranking_order"] = llm_ranking
            
            # Reorder the answers to match the shuffled order for later use
            line["answers"] = list(shuffled_answers)
            
            return line
    
    # If not successful or parsing failed, return None to discard the line
    return None

def verify_answers_api(lines, args):
    """Verifies if the API model can correctly rank the selected answers."""
    new_lines = []
    
    pool_fn = partial(verify_line_api, model=args.model_path, temperature=args.temperature)

    if args.multi_process == "False":
        for line in tqdm.tqdm(lines):
            new_line = verify_line_api(line, model=args.model_path, temperature=args.temperature)
            if new_line:
                new_lines.append(new_line)
    else:
        processed_lines = pool.map(pool_fn, lines)
        new_lines = [line for line in processed_lines if line is not None]
    
    return new_lines

def verify_answers_vllm(lines, args):
    """Verifies if the vLLM model can correctly rank the selected answers."""
    llm = vllm.LLM(model=args.model_path, tensor_parallel_size=torch.cuda.device_count())
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    sampling_params = vllm.SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    
    prompts = []
    shuffled_data_map = [] 

    for line in lines:
        if len(line.get("answers", [])) != 3:
            continue
        
        # The ground truth order is already correct > other > incorrect, so we only need to shuffle and check
        indexed_answers = list(enumerate(line["answers"]))
        random.shuffle(indexed_answers)
        shuffled_indices, shuffled_answers = zip(*indexed_answers)

        prompt_content = VERIFICATION_RANKING_PROMPT.format(
            question=line["question"],
            reference=line["reference"],
            answer1=shuffled_answers[0]['answer'],
            answer2=shuffled_answers[1]['answer'],
            answer3=shuffled_answers[2]['answer'],
        )
        messages = [{"role": "user", "content": prompt_content}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        shuffled_data_map.append({'line': line, 'shuffled_indices': shuffled_indices, 'shuffled_answers': shuffled_answers})

    if not prompts:
        return []

    outputs = llm.generate(prompts, sampling_params)
    
    output_results = []
    for i, output in enumerate(outputs):
        original_data = shuffled_data_map[i]
        line = original_data['line']
        shuffled_indices = original_data['shuffled_indices']
        shuffled_answers = original_data['shuffled_answers']
        response = output.outputs[0].text.strip()
        
        llm_ranking = parse_evaluation_ranking(response)
        line["verification_reason"] = response

        if llm_ranking and len(llm_ranking) == 3:
            ranked_ground_truth_indices = [shuffled_indices[i - 1] for i in llm_ranking]
            is_successful = (ranked_ground_truth_indices == [0, 1, 2])
            if is_successful:
                line["verification_successful"] = True
                line["llm_ranking_order"] = llm_ranking
                # Reorder the answers to match the shuffled order for later use
                line["answers"] = list(shuffled_answers)
                output_results.append(line)
        # If not successful, simply skip appending to output_results
    
    return output_results

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
    parser.add_argument("--phase", type=str, choices=("filtering", "generation", "evaluation", "selection", "selection-train", "verification"))
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
            try:
                line = json.loads(line.strip())
            except:
                print("An abnormal line in jsonl file!")
                continue
            lines.append(line)
            
    # For controlling test data number    
    # if len(lines) > 2000:
    #     random.shuffle(lines)
    #     lines = lines[:2000]

    if args.phase == "filtering":
        lines = instruction_filtering(lines, args)
    elif args.phase == "generation":
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
            lines = verify_answers_vllm(lines, args)
        elif args.model_type == "api":
            lines = verify_answers_api(lines, args)

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for line in lines:
            fout.write(json.dumps(line)+"\n")
