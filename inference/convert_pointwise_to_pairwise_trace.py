import json
from argparse import ArgumentParser
from transformers import AutoTokenizer

parser = ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--tokenizer-path", type=str, default="/workspace/HFModels/Qwen2.5-7B-Instruct")

args = parser.parse_args()

verdict_prompt = """You are an expert fact-checking assistant. Your task is to determine whether the answer is factually correct or not.

Question: {question}
Answer: {answer}
Retrieved Reference Information: {formatted_facts}

Based on the question, answer, and the reference information retrieved before, determine if the answer is factually correct or not."""

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

with open(args.input_file, "r", encoding="utf-8") as fin,\
open(args.output_file, "w", encoding="utf-8") as fout:
    lines = [json.loads(line.strip()) for line in fin.readlines()]
    for i, line in enumerate(lines):
        if line["extracted_facts"] == []:
            formatted_facts = "No useful information retrieved."
        else:
            formatted_facts = " ".join([f"{i}. {fact}" for i,fact in enumerate(line["extracted_facts"])])

        prompt = verdict_prompt.format(question=line["question"], answer=line["answer"]["answer"], formatted_facts=formatted_facts)
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)

        if i % 2 == 0:
            positive_trace = prompt
        else:
            negative_trace = prompt

            new_line = {
                "pos_trace": positive_trace,
                "neg_trace": negative_trace,
                "verify_result": 0,
            }
            fout.write(json.dumps(new_line)+"\n")