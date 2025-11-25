import json

with open("dev.jsonl", "r", encoding="utf-8") as fin,\
open("nq_test_gen.jsonl", "w", encoding="utf-8") as fout:

    lines = [json.loads(line) for line in fin.readlines()]
    for line in lines:
        new_line = {"question": line["question"], "reference": line["golden_answers"]}
        fout.write(json.dumps(new_line)+"\n")