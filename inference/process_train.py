import json
from argparse import ArgumentParser
from transformers import AutoTokenizer
from utils_prompts import CLS_VERDICT_PROMPT

parser = ArgumentParser()
parser.add_argument("--input-file", type=str, required=True)
parser.add_argument("--output-prefix", type=str, required=True)
parser.add_argument("--cls-input", type=str, default="facts", choices=("trace", "facts", "nothing"))
parser.add_argument("--tokenizer-path", type=str, default="/workspace/HFModels/Qwen2.5-7B-Instruct")

args = parser.parse_args()

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    output_file = args.output_prefix + "-" + args.mode + ".json"
    with open(args.input_file, "r", encoding="utf-8") as fin,\
    open(output_file, "w", encoding="utf-8") as fout:
        lines = [json.loads(line.strip()) for line in fin.readlines()]

        verdict_prompt = CLS_VERDICT_PROMPT

        for i, line in enumerate(lines):

            all_prompts = []
            for answer in line["answers"]:
                
                assert len(line["answers"]) == 2

                if answer["extracted_facts"] == []:
                    formatted_facts = "No useful information retrieved."
                else:
                    formatted_facts = " ".join([f"{i+1}. {fact}" for i,fact in enumerate(answer["extracted_facts"])])

                prompt = verdict_prompt.format(question=line["question"], answer=answer["answer"], formatted_facts=formatted_facts)
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)

                all_prompts.append(prompt)
            
            if line.get("verify_result") == 0:
                positive_trace = all_prompts[0]
                negative_Trace = all_prompts[1]
            elif line.get("verify_result") == 1:
                positive_trace = all_prompts[1]
                negative_Trace = all_prompts[0]
            else:
                raise Exception("Please check your verify_result!")
            
            new_line = {
                **line,
                "pos_trace": positive_trace,
                "neg_trace": negative_Trace,
            }

            fout.write(json.dumps(new_line) + "\n")