from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import argparse
import json
import tqdm

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
)

def evaluate_final_results(y_true, y_pred):
    """
    计算并打印评估指标。
    """

    metrics_dict = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "evaluated_count": len(y_true),
    }

    print("\n--- Evaluation Results ---")
    for key, value in metrics_dict.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("--------------------------\n")
    return metrics_dict

def remove_final_verdict(text):
    verdict_start = text.rfind("<answer>") #text.rfind("Therefore, the final verdict is: <answer>")
    if verdict_start != -1:
        return text[:verdict_start].strip()
    return text   

# --- 示例用法 ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        # default="/home/huanghui/models/Qwen_Qwen2.5-7B-Instruct",
        help="Path to the SGLang-compatible model.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        # default="/home/huanghui/Search-R1/fact_checking_dataset/bamboogle_test.jsonl",
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        # default="/home/huanghui/Search-R1/results/bamboogle_test-Qwen_Qwen2.5-7B-Instruct-local_retrieval.jsonl",
        help="Path to save the output JSONL file.",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        choices=("pairwise", "pointwise"),
        default="pairwise",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).cuda()

    with open(args.input_file, "r", encoding="utf-8") as fin:
        lines = [json.loads(line) for line in fin.readlines()]
        predictions = []
        labels = []
        i = 0
        for line in tqdm.tqdm(lines):
            if "model_output_trace" not in line.keys():
                inputs = tokenizer.apply_chat_template([
                    {"role": "user", "content": line["question"]},
                    {"role": "assistant", "content": line["response"]},
                ], tokenize=False)
                inputs = tokenizer(inputs, return_tensors="pt").to(model.device)
            else:
                model_output_trace = remove_final_verdict(line["model_output_trace"])
                inputs = tokenizer(model_output_trace, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**inputs)

            prediction = outputs.logits.item()

            if i % 2 == 0:
                positive_prediction = prediction
            else:
                negative_prediction = prediction

                if positive_prediction > negative_prediction:
                    predictions.append(0)
                else:
                    predictions.append(1)

                labels.append(0)
                
            i += 1

        accuracy = evaluate_final_results(predictions, labels)