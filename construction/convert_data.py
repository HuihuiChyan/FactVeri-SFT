import json
import random

DATA="nq_hotpot_train_head"
INPUT = f"./{DATA}/{DATA}_pointwise.jsonl"
OUTPUT = F"./{DATA}/{DATA}_pairwise.jsonl"

random.seed(42)

with open(INPUT, "r", encoding="utf-8") as fin,\
open(OUTPUT, "w", encoding="utf-8") as fout:
    lines = [json.loads(line.strip()) for line in fin.readlines()]
    new_lines = []
    for i,line in enumerate(lines):
        if i % 2 == 0:
            positive_response = line["response"]
            assert line["label"] == "supported"
        elif i % 2 == 1:
            negative_response = line["response"]
            question = line["question"]
            assert line["label"] == "unsupported"

            if random.random() > 0.5:
                new_line = {
                    "question": question,
                    "label": "response1",
                    "response1": positive_response,
                    "response2": negative_response,
                }
            else:
                new_line = {
                    "question": question,
                    "label": "response2",
                    "response1": negative_response,
                    "response2": positive_response,
                }                
            fout.write(json.dumps(new_line)+"\n")