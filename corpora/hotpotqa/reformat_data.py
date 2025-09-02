import json
INPUT = "hotpotqa_pairwise.jsonl"
OUTPUT = "hotpotqa_testset_pairwise.jsonl"
OUTPUT_POINTWISE = "hotpotqa_testset_pointwise.jsonl"
with open(INPUT, "r", encoding="utf-8") as fin,\
open(OUTPUT, "w", encoding="utf-8") as fout,\
open(OUTPUT_POINTWISE, "w", encoding="utf-8") as fout_pointiwse:
    lines = [json.loads(line.strip()) for line in fin.readlines()]
    for line in lines:
        new_line = {
                        "question": line["question"], 
                        "answer1": {"answer": line["response1"]},
                        "answer2": {"answer": line["response2"]},
                        "verify_result": int(line["label"] != "response1"),
                    }
        fout.write(json.dumps(new_line)+"\n")

        if line["label"] == "response1":

            new_line_pointwise = {
                                    "question": line["question"], 
                                    "answer": {"answer": line["response1"]},
                                    "verify_result": 1,
                                }
            fout_pointiwse.write(json.dumps(new_line_pointwise)+"\n")
            new_line_pointwise = {
                                    "question": line["question"], 
                                    "answer": {"answer": line["response2"]},
                                    "verify_result": 0,
                                }
            fout_pointiwse.write(json.dumps(new_line_pointwise)+"\n")
        
        else:

            new_line_pointwise = {
                                    "question": line["question"], 
                                    "answer": {"answer": line["response2"]},
                                    "verify_result": 1,
                                }
            fout_pointiwse.write(json.dumps(new_line_pointwise)+"\n")
            new_line_pointwise = {
                                    "question": line["question"], 
                                    "answer": {"answer": line["response1"]},
                                    "verify_result": 0,
                                }
            fout_pointiwse.write(json.dumps(new_line_pointwise)+"\n")