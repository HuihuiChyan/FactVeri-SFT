import argparse
import json
import re
import os
import tqdm
from typing import List, Dict

# Import sglang libraries
import sglang as sgl

def parse_args():
    """Parses command-line arguments for the pointwise summarization script."""
    parser = argparse.ArgumentParser(
        description="Pointwise fact-checking summarization script that processes output from infer_batch_sglang.py"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, 
        help="Path to the SGLang-compatible model for verdict generation."
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Path to the input JSONL file (output from infer_batch_sglang.py in pointwise mode)."
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Path to save the output JSONL file with verdicts and verification results."
    )
    parser.add_argument(
        "--disable_thinking", action="store_true", default=False,
        help="Disable the model's thinking process."
    )
    parser.add_argument(
        "--max_token", type=int, default=2048, 
        help="Maximum new tokens to generate for verdicts."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, 
        help="Sampling temperature for verdict generation."
    )
    parser.add_argument(
        "--batch_size", type=int, default=100,
        help="Batch size for processing verdicts."
    )
    return parser.parse_args()

def extract_structured_response(model_generated_output: str) -> Dict[str, str]:
    """Extracts Useful Facts, Reasoning, and Final Verdict from the model's output."""
    if not isinstance(model_generated_output, str):
        return {"useful_facts": "", "reasoning": "", "final_verdict": "Invalid"}

    # 统一设置 flags，re.DOTALL 让 '.' 可以匹配换行符，这对于多行内容至关重要
    flags = re.IGNORECASE | re.DOTALL

    # 提取 Useful Facts (匹配到 "Reasoning:"、"Final Verdict:" 或字符串结尾)
    facts_match = re.search(r"\**Useful Facts:\**\s*(.*?)\s*(?=\**Reasoning:|\**Final Verdict:|$)", model_generated_output, flags)
    facts = facts_match.group(1).strip() if facts_match else ""

    # 提取 Reasoning (匹配到 "Final Verdict:" 或字符串结尾)
    reasoning_match = re.search(r"\**Reasoning:\**\s*(.*?)\s*(?=\**Final Verdict:|$)", model_generated_output, flags)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    # 提取 Final Verdict (匹配 "Final Verdict:" 之后的所有内容)
    verdict_match = re.search(r"\**Final Verdict:\**\s*(.*)", model_generated_output, flags)
    
    verdict_text = verdict_match.group(1).strip().lower() if verdict_match else ""
    
    # Normalize the verdict
    final_verdict = "Invalid"
    if "correct" in verdict_text and "incorrect" not in verdict_text:
        final_verdict = "Correct"
    elif "incorrect" in verdict_text:
        final_verdict = "Incorrect"
    elif "intermediate" in verdict_text:
        final_verdict = "Intermediate"

    return {"useful_facts": facts, "reasoning": reasoning, "final_verdict": final_verdict}

def batched_sglang_generation(input_ids, sampling_params, engine, batch_size=100):
    """Generates text in batches using the SGLang engine."""
    batched_input_ids = [input_ids[i:i + batch_size] for i in range(0, len(input_ids), batch_size)]
    results = []
    for batch in tqdm.tqdm(batched_input_ids, desc="Generating Verdicts"):
        results.extend(engine.generate(input_ids=batch, sampling_params=sampling_params))
    return results

def evaluate_final_results_pointwise(results: List[Dict]):
    """Calculates and prints binary classification metrics for the pointwise scheme."""
    tp, tn, fp, fn, total_answers, invalid_predictions = 0, 0, 0, 0, 0, 0
    
    for item in results:
        for answer in item.get("answers", []):
            total_answers += 1
            gt = answer.get("verify_result", "").lower()
            fv = answer.get("final_verdict", "").lower()
            
            if fv == "invalid": 
                invalid_predictions += 1
                continue
                
            # Normalize ground truth and prediction
            gt_pos = gt in ["correct", "intermediate", "irrelevant"]
            pred_pos = fv in ["correct", "intermediate", "irrelevant"]
            
            if gt_pos and pred_pos: 
                tp += 1
            elif not gt_pos and not pred_pos: 
                tn += 1
            elif not gt_pos and pred_pos: 
                fp += 1
            elif gt_pos and not pred_pos: 
                fn += 1
    
    # Calculate metrics
    accuracy = (tp + tn) / total_answers if total_answers > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    invalid_ratio = invalid_predictions / total_answers if total_answers > 0 else 0
    
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "invalid_ratio": round(invalid_ratio, 4),
        "total_answers": total_answers,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }
    
    print("\n--- Pointwise Evaluation Results ---")
    print(f"Total Answers: {total_answers}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Invalid Predictions Ratio: {invalid_ratio:.4f}")
    print(f"Confusion Matrix - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print("------------------------------------")
    
    return metrics

def verify_judgment_correctness(results: List[Dict]):
    """Verifies the correctness of judgments by comparing with ground truth."""
    correct_judgments = 0
    total_judgments = 0
    
    for item in results:
        for answer in item.get("answers", []):
            if "verify_result" in answer and "final_verdict" in answer:
                total_judgments += 1
                gt = answer["verify_result"].lower()
                pred = answer["final_verdict"].lower()
                
                # Normalize for comparison
                gt_normalized = "correct" if gt in ["correct", "intermediate"] else "incorrect"
                pred_normalized = "correct" if pred in ["correct", "intermediate"] else "incorrect"
                
                if gt_normalized == pred_normalized:
                    correct_judgments += 1
    
    accuracy = correct_judgments / total_judgments if total_judgments > 0 else 0
    print(f"\n--- Judgment Verification ---")
    print(f"Correct Judgments: {correct_judgments}/{total_judgments}")
    print(f"Judgment Accuracy: {accuracy:.4f}")
    print("----------------------------")
    
    return {"judgment_accuracy": accuracy, "correct_judgments": correct_judgments, "total_judgments": total_judgments}

def format_conversation_history(retrieval_path: List[Dict]) -> str:
    """Formats the retrieval path into a readable string."""
    formatted_parts = []
    turn_counter = 1
    # Skip the initial user prompt which is just a setup
    for message in retrieval_path[1:]:
        role = message.get("role")
        content = message.get("content", "")
        
        if role == "assistant":
            formatted_parts.append(f"--- Search Turn {turn_counter} ---")
            formatted_parts.append(f"[Model's Thought Process and Action]\n{content.strip()}")
            turn_counter += 1
        elif role == "tool":
            formatted_parts.append(f"[Search Result]\n{content.strip()}")
            
    return "\n\n".join(formatted_parts)

def process_pointwise_factuality_judgment(args):
    """
    Main function to process pointwise factuality judgment.
    Reads output from infer_batch_sglang.py, generates verdicts, and evaluates results.
    """
    # Initialize SGLang engine
    print("Initializing SGLang engine...")
    engine = sgl.Engine(model_path=args.model_path, trust_remote_code=True, mem_fraction_static=0.8)
    tokenizer = engine.tokenizer_manager.tokenizer
    
    # Load input data
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f if line.strip()]
    
    print(f"Loaded {len(input_data)} items from input file.")
    
    # Define the verdict prompt template
    POINTWISE_VERDICT_PROMPT = """You are a fact-checking expert. Below is a history of search actions performed by an agent to gather information about a specific question and answer. Your task is to analyze this history and determine the factual correctness of the answer.

### SEARCH HISTORY ###
{search_history}

### VERIFICATION TASK ###
Based on the information from the search history, please verify the following:

**Question:** {question}
**Answer to Verify:** {answer}

### ANALYSIS AND VERDICT ###
Based on your analysis, provide a structured response in the following format. Do not add any other text outside this structure.

**Useful Facts:** [List key facts from the search history relevant to the answer, separated by semicolons. Example: Fact1; Fact2; Fact3;]
**Reasoning:** [Provide a step-by-step reasoning for your verdict based on the useful facts.]
**Final Verdict:** [Your verdict: Correct, Incorrect, or Intermediate]"""

    # Prepare input for verdict generation
    pointwise_input_ids = []
    response_map = []
    
    for i, item in enumerate(input_data):
        question = item.get("question", "")
        retrieval_path = item.get("retrieval_path", [])
        
        formatted_history = format_conversation_history(retrieval_path)
        
        for ans_idx, answer_obj in enumerate(item.get("answers", [])):
            answer_text = answer_obj.get("answer", "")
            
            # Create the final prompt content as a single string
            prompt_content = POINTWISE_VERDICT_PROMPT.format(
                search_history=formatted_history,
                question=question,
                answer=answer_text
            )
            
            # Wrap the formatted prompt content into a single user message
            final_messages = [{"role": "user", "content": prompt_content}]
            
            # Tokenize the conversation using the chat template
            tokenized_prompt = tokenizer.apply_chat_template(
                conversation=final_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=(not args.disable_thinking)
            )
            
            pointwise_input_ids.append(tokenized_prompt)
            response_map.append({"item_index": i, "answer_index": ans_idx})
    
    if not pointwise_input_ids:
        print("No answers found to generate verdicts for.")
        return

    print(f"Generating verdicts for {len(pointwise_input_ids)} individual answers...")
    
    # Generate verdicts in batches
    sampling_params = {
        "max_new_tokens": args.max_token, 
        "temperature": args.temperature
    }
    
    verdict_responses = batched_sglang_generation(
        input_ids=pointwise_input_ids, 
        sampling_params=sampling_params, 
        engine=engine,
        batch_size=args.batch_size
    )
    
    # Process responses and add verdicts to the data
    print("Processing verdict responses...")
    for i, response in enumerate(verdict_responses):
        mapping = response_map[i]
        item_index, answer_index = mapping["item_index"], mapping["answer_index"]
        generated_text = response["text"]
        
        # Extract structured data and add to the answer object
        extracted_data = extract_structured_response(generated_text)
        
        input_data[item_index]["answers"][answer_index].update({
            "verdict_response": generated_text,
            "useful_facts": extracted_data["useful_facts"],
            "reasoning": extracted_data["reasoning"],
            "final_verdict": extracted_data["final_verdict"],
        })
    
    # Save results with verdicts
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for result in input_data:
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Results with verdicts saved to {args.output_file}")
    
    # Evaluate the results
    print("Evaluating results...")
    evaluation_metrics = evaluate_final_results_pointwise(input_data)
    
    # Verify judgment correctness
    verification_results = verify_judgment_correctness(input_data)
    
    # Print summary
    print(f"\n--- Summary ---")
    print(f"Input file: {os.path.basename(args.input_file)}")
    print(f"Output file: {os.path.basename(args.output_file)}")
    print(f"Model: {os.path.basename(args.model_path)}")
    print(f"Total items processed: {len(input_data)}")
    print(f"Total answers processed: {evaluation_metrics.get('total_answers', 0)}")
    print(f"Overall F1 Score: {evaluation_metrics.get('f1', 0):.4f}")
    print(f"Judgment Accuracy: {verification_results.get('judgment_accuracy', 0):.4f}")
    print("---------------")
    
    return {
        "evaluation_metrics": evaluation_metrics,
        "verification_results": verification_results,
        "processed_data": input_data
    }

def main():
    """Main execution function."""
    args = parse_args()
    
    try:
        results = process_pointwise_factuality_judgment(args)
        print("Processing completed successfully!")
        return results
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()