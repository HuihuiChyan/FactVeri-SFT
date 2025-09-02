import argparse
import json
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
)
from typing import List, Dict
from process_cls_input import create_prompt_for_cls

def evaluate_final_results(results: List[Dict]):
    """
    Calculates and prints evaluation metrics for a multi-class (best-of-N) scenario.
    """
    y_true, y_pred = [], []

    for item in results:
        # The ground truth label from the dataset
        true_label = item.get("verify_result") 
        # The prediction made by our classifier
        pred_label = item.get("prediction")

        if true_label is None:
            continue  # Skip items without a ground truth label

        y_true.append(true_label)
        y_pred.append(pred_label)

    if not y_true:
        print("Evaluation failed. No valid ground truth labels found.")
        return None

    # Use 'macro' average for multi-class precision, recall, and F1
    metrics_dict = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "evaluated_count": len(results),
    }

    print("\n--- Evaluation Results ---")
    for key, value in metrics_dict.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("--------------------------\n")
    return metrics_dict

def main():
    """Main function to run classification and evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate fact-checking results using a classifier on extracted facts."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the sequence classification model.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--cls-input",
        type=str,
        choices=("facts", "naive", "facts_trace"),
        default="naive"
    )
    args = parser.parse_args()

    print("Loading classifier model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).cuda()
    model.eval() # Set model to evaluation mode

    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as fin:
        lines = [json.loads(line) for line in fin.readlines()]

    final_results = []
    with torch.no_grad():
        for item in tqdm.tqdm(lines, desc="Classifying Answers"):
            answers = item["answers"]
            answer_scores = []

            for i, answer_item in enumerate(answers):
                
                # Apply the chat template to format the input as a conversation
                prompt = create_prompt_for_cls(item, answer_item, args.cls_input)
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
                
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
                
                # Get the raw logit score from the classifier
                score = model(**inputs).logits.item()
                answer_scores.append(score)
                # Optionally save the score for each answer
                answer_item["factuality_score"] = score

            # The prediction is the index of the answer with the highest score
            prediction = answer_scores.index(max(answer_scores))
            item["prediction"] = prediction
            final_results.append(item)

    print(f"Saving results with predictions to {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for result in final_results:
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("Starting final evaluation...")
    metrics = evaluate_final_results(final_results)
    if metrics:
        print(f"Final evaluation metrics: {json.dumps(metrics, indent=4)}")

if __name__ == "__main__":
    main()
