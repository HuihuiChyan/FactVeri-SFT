import json

with open("science_qa_combination-Qwen2.5-7B-Instruct-direct_gen-pairwise.jsonl", "r") as fin:
    lines1 = [json.loads(line.strip()) for line in fin.readlines()]

with open("science_qa_combination-Qwen2.5-7B-Instruct-local_retrieval-pairwise.jsonl", "r") as fin:
    lines2 = [json.loads(line.strip()) for line in fin.readlines()]

with open("temp.jsonl", "w") as fout:
    for line1, line2 in zip(lines1, lines2):
        if (line1['verify_result'] != line1['final_verdict'] and line2['verify_result'] == line2['final_verdict']) \
            or (line1['verify_result'] == line1['final_verdict'] and line2['verify_result'] != line2['final_verdict']):

            new_line = {
                "answer1": line1["answer1"],
                "answer2": line2["answer2"],
                "verify_result": line1["verify_result"],
                "model_output_trace_no_rag": line1["model_output_trace"],
                "model_output_trace_wt_rag": line2["model_output_trace"],
            }
            fout.write(json.dumps(new_line)+"\n")