import argparse
import json
from utils_prompts import POINTWISE_VERDICT_PROMPT
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--input-file", type=str)
parser.add_argument("--tokenizer-path", type=str, default="/workspace/HFModels/Qwen3-4B")
parser.add_argument("--output-file-train", type=str)
parser.add_argument("--output-file-test", type=str)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

with open(args.input_file, "r", encoding="utf-8") as fin:
    lines = [json.loads(line.strip()) for line in fin.readlines()]
    all_data = []
    for line in tqdm(lines):
        for answer in line["answers"]:
            prompt_content = POINTWISE_VERDICT_PROMPT.format(question=line["question"], answer=answer["answer"])
            prompt = answer['retrieval_path'] + [{"role": "user", "content": prompt_content}]
            prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
            prompt = [{"role": "user", "content": prompt[17:-11]}]
            data = {
                "data_source": "fact-checking",
                "prompt": prompt,
                "ability": "fact-checking",
                "reward_model": {"style": "rule", "ground_truth": answer["verfy_result"]},
                "extra_info": {
                    "split": "train",
                    "answer": answer["verfy_result"],
                },
            }
            all_data.append(data)

    df_train = pd.DataFrame(all_data[10:])
    df_test = pd.DataFrame(all_data[:10])
    
    # Write the DataFrame to a Parquet file.
    # `pyarrow` is used as the engine by default.
    # `index=False` prevents pandas from writing the DataFrame index as a column.
    df_train.to_parquet(args.output_file_train, index=False)
    df_test.to_parquet(args.output_file_test, index=False)