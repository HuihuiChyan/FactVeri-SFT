import argparse
import json
import re
import os
import tqdm
import sklearn
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
        batched_results = engine.generate(input_ids=batch, sampling_params=sampling_params)
        results.extend(batched_results)
            
    return results

def evaluate_final_results_pointwise(results: List[Dict]):
    """
    根据新的多分类规则（合并intermediate/irrelevant）计算和打印评估指标。
    
    规则:
    1. 将 "irrelevant" 和 "intermediate" 标签合并为 "intermediate" 类。
    2. 计算 "correct", "intermediate", "incorrect" 这三个类别的 Acc 和 F1。
    3. 将空字符串或其他意外标签视为 "incorrect"。
    """
    
    ground_truths = []
    predictions = []
    total_answers = 0
    invalid_predictions = 0
    
    # 定义标签列表，用于 sklearn 报告
    class_labels = ["correct", "intermediate", "incorrect"]

    def normalize_label(label: str) -> str:
        """根据规则合并标签"""
        label = label.lower()
        if label in ["intermediate", "irrelevant"]:
            return "intermediate"
        elif label in ["correct", "incorrect"]:
            return label
        else:
            # 将空字符串或其他意外值视为 "incorrect"
            return "incorrect"

    for item in results:
        for answer in item.get("answers", []):
            total_answers += 1
            
            # 1. 提取 Ground Truth (gt)，并处理拼写错误
            if "verify_result" in answer.keys():
                gt_raw = answer.get("verify_result", "")
            else:
                gt_raw = answer.get("verfy_result", "")  # 处理可能的拼写错误
            
            # 2. 提取 Final Verdict (fv)
            fv_raw = answer.get("final_verdict", "")
            
            # 3. 跳过 "invalid" 的预测
            if fv_raw.lower() == "invalid": 
                invalid_predictions += 1
                continue
                
            # 4. 规范化标签并添加到列表中
            gt_norm = normalize_label(gt_raw)
            fv_norm = normalize_label(fv_raw)
            
            ground_truths.append(gt_norm)
            predictions.append(fv_norm)
    
    # --- 计算指标 ---
    total_valid = len(ground_truths)
    
    if total_valid == 0:
        # 处理没有有效答案的情况
        accuracy = 0.0
        f1_macro = 0.0
        
    # 1. 计算 Accuracy
    # 这完全符合你的要求：(gt==fv) 或 (gt,fv 都在 {intermediate, irrelevant})
    accuracy = sklearn.metrics.accuracy_score(ground_truths, predictions)
    
    # 2. 计算 F1-Score (Macro 和 Weighted)
    f1_macro = sklearn.metrics.f1_score(ground_truths, predictions, average='macro', labels=class_labels, zero_division=0)
    invalid_ratio = invalid_predictions / total_answers if total_answers > 0 else 0
    
    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "total_answers": total_answers,
        "invalid_answers": invalid_predictions,
        "invalid_ratio": round(invalid_ratio, 4),
    }
    
    # --- 打印结果 ---
    print("\n--- Pointwise Evaluation Results ------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Total Answers: {total_answers}")
    print(f"Invalid Answers: {invalid_predictions} ({invalid_ratio:.4f})")
    print("-----------------------------------------------")
    
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
        
        for ans_idx, answer_obj in enumerate(item.get("answers", [])):

            retrieval_path = answer_obj.get("retrieval_path", [])
            formatted_history = format_conversation_history(retrieval_path)

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
        "temperature": args.temperature,
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
        
        verdict = {
            "verdict_response": generated_text,
            "useful_facts": extracted_data["useful_facts"],
            "reasoning": extracted_data["reasoning"],
            "final_verdict": extracted_data["final_verdict"],
        }
        input_data[item_index]["answers"][answer_index].update(verdict)
    
    # Save results with verdicts
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for result in input_data:
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Results with verdicts saved to {args.output_file}")
    
    # Evaluate the results
    print("Evaluating results...")
    evaluation_metrics = evaluate_final_results_pointwise(input_data)

    return {
        "evaluation_metrics": evaluation_metrics,
        "processed_data": input_data
    }

if __name__ == "__main__":
    args = parse_args()
    
    results = process_pointwise_factuality_judgment(args)
    print("Processing completed successfully!")
