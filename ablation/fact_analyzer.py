import openai
import time
import json
import concurrent.futures
from collections import defaultdict
from multiprocessing import Pool, Value, Manager
import os
import sys
import threading
import logging
from typing import List, Dict # Added for type hinting in the new function
from functools import partial # Added for multiprocessing

from scipy.stats import kendalltau
import timeout_decorator
import re

# Set up a global counter and start time for the user's original logging style
counter = Value('i', 0)
start_time = time.time()

def request_gpt(messages: list, model: str, temperature: float) -> str:
    """
    Makes a request to the OpenAI-compatible API.
    Uses environment variables for credentials, with fallbacks.
    """
    # Use env variables first, fallback to hardcoded values
    api_key = os.getenv("OPENAI_API_KEY", "sk-agEcX3Su78Bu09c2F49978C6Ba424977B936C8710fAb42E0")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.shubiaobiao.cn/v1/")

    if not api_key:
        raise ValueError("API key not found. Please set OPENAI_API_KEY or check the script.")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    max_tries = 3
    res = ''
    for i in range(max_tries):
        try:
            chat_completion = client.chat.completions.create(
                model=payload['model'],
                temperature=payload['temperature'],
                messages=payload['messages']
            )
            res = chat_completion.choices[0].message.content
            break  # Exit loop on success
        except Exception as e:
            print(f"API Error (Attempt {i+1}/{max_tries}): {e}", file=sys.stderr)
            time.sleep(5)
            continue

    # --- Original counter logic from user's function ---
    with counter.get_lock():
        counter.value += 1
        count_val = counter.value
    
    if count_val % 10 == 0: # Log every 10 calls
        avg_time = round((time.time() - start_time) / count_val, 2)
        print(f"API call {count_val} finished! {avg_time}s avg per call.", file=sys.stderr)
    # --- End of counter logic ---

    if not res:
        print(f"API call failed after {max_tries} tries.", file=sys.stderr)

    return res

# --- Initialization function for multiprocessing.Pool ---
def init_worker(c, t):
    """Initializes worker process for multiprocessing pool with shared variables."""
    global counter
    global start_time
    counter = c
    start_time = t

# --- New Functions for Analysis ---

def classify_usefulness(question: str, answer: str, useful_facts: str) -> Dict[str, any]:
    """
    Calls the LLM to classify if useful_facts are helpful.
    Returns a dictionary: {"is_useful": bool, "response_text": str}
    """
    # Updated system prompt
    system_prompt = "You are an evaluator. Your task is to determine if a set of 'Useful Facts' is helpful for verifying if an 'Answer' correctly responds to a 'Question'. Please first provide your explanation, and then conclude your answer in a new line with only 'Yes' or 'No'."
    
    user_prompt = f"""Question: {question}
Answer: {answer}
Useful Facts: {useful_facts}

Are the 'Useful Facts' helpful for verifying the 'Answer' in relation to the 'Question'?
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Use a fast and cheap model for classification
    model = "gpt-4o"
    temperature = 0.0

    try:
        response = request_gpt(messages, model, temperature)
        if not response:
            print(f"Warning: Received empty response from LLM.", file=sys.stderr)
            return {"is_useful": False, "response_text": "Error: Empty response from LLM."}

        # New parsing logic:
        # Find the last non-empty line of the response.
        lines = response.strip().split('\n')
        last_line_cleaned = ""
        if lines:
            for line in reversed(lines):
                # Clean up punctuation, markdown, etc.
                # This removes anything that's not a word character or whitespace
                cleaned = re.sub(r'[^\w\s]', '', line, re.UNICODE).strip().lower()
                if cleaned:
                    last_line_cleaned = cleaned
                    break

        # Check 1: Exact match on the cleaned last line
        if last_line_cleaned == "yes":
            return {"is_useful": True, "response_text": response}
        if last_line_cleaned == "no":
            return {"is_useful": False, "response_text": response}
            
        # Check 2: Check for "yes" or "no" as the *only* word in a potentially longer last line
        # (e.g., "conclusion yes", "the answer is no")
        words = set(last_line_cleaned.split())
        
        has_yes = "yes" in words
        has_no = "no" in words

        # If the last line contains "yes" but not "no"
        if has_yes and not has_no:
            return {"is_useful": True, "response_text": response}
        # If the last line contains "no" but not "yes"
        if has_no and not has_yes:
            return {"is_useful": False, "response_text": response}

        # Check 3: Fallback if the last line is ambiguous (has both) or has neither.
        # Search the *entire* response for the *last* standalone "yes" or "no".
        # This handles cases where the LLM might have forgotten the "new line" instruction.
        response_lower = response.strip().lower()
        # Find all matches for standalone "yes" or "no"
        matches = list(re.finditer(r'\b(yes|no)\b', response_lower))
        
        if matches:
            # Get the last match
            last_match = matches[-1].group(0)
            if last_match == "yes":
                return {"is_useful": True, "response_text": response}
            if last_match == "no":
                return {"is_useful": False, "response_text": response}

        # If all parsing fails (e.g., response is just "The facts are correct."), warn and return False.
        warning_msg = f"Warning: Could not parse 'Yes' or 'No' from LLM response: '{response}'"
        print(warning_msg, file=sys.stderr)
        return {"is_useful": False, "response_text": f"{warning_msg}\n\n{response}"}
        
    except Exception as e:
        error_msg = f"Error in classify_usefulness: {e}"
        print(error_msg, file=sys.stderr)
        return {"is_useful": False, "response_text": error_msg}

def get_score_ranking_order(scores: list) -> list:
    """
    Takes a list of scores and returns a list of original 1-based indices,
    ordered from highest score to lowest score.
    e.g., [0.3, 0.5, 0.4] -> [2, 3, 1]
    """
    # Enumerate to get (index, score) tuples
    score_tuples = list(enumerate(scores))
    # Sort by score (item[1]) in descending order
    score_tuples_sorted = sorted(score_tuples, key=lambda x: x[1], reverse=True)
    # Return 1-based original indices
    return [i + 1 for i, s in score_tuples_sorted]

def get_gt_ranking_order(ranks: list) -> list:
    """
    Takes a list of ranks (1-indexed) and returns a list of original
    indices, ordered from best rank (1) to worst rank.
    e.g., [2, 1, 3] -> [1, 0, 2]
    """
    # Enumerate to get (index, rank) tuples
    rank_tuples = list(enumerate(ranks))
    # Sort by rank (item[1]) in ascending order
    rank_tuples_sorted = sorted(rank_tuples, key=lambda x: x[1])
    # Return just the original indices
    return [i for i, r in rank_tuples_sorted]

def process_sample(line_data, sample_index):
    """
    Processes a single sample (line) from the JSONL.
    Calls the LLM 3 times (once for each answer) in parallel.
    Returns the modified line_data (with category and predicted_ranking added).
    """
    question = line_data['question']
    answers_data = line_data['answers']

    yes_count = 0
    # This list will store booleans in the correct order for logging
    ordered_bool_results = [False] * len(answers_data)
    
    # Run the 3 classifications in parallel for this single sample
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Map each future to the index of the answer it's processing
        futures_map = {} # {future: index}
        for i, ans_data in enumerate(answers_data):
            future = executor.submit(
                classify_usefulness,
                question,
                ans_data.get('answer', ''), # Use .get for safety
                ans_data.get('useful_facts', '')
            )
            futures_map[future] = i # Key is future, value is index (0, 1, or 2)
        
        # Process as they complete
        for future in concurrent.futures.as_completed(futures_map):
            index = futures_map[future] # Get the index (0, 1, or 2)
            try:
                # This is now a dict: {"is_useful": bool, "response_text": str}
                result_dict = future.result() 
                
                is_useful = result_dict["is_useful"]
                response_text = result_dict["response_text"]
                
                # Add the full response text to the correct answer object
                line_data['answers'][index]['usefulness_analysis'] = response_text
                
                if is_useful:
                    yes_count += 1
                
                ordered_bool_results[index] = is_useful # Store boolean in the correct slot

            except Exception as e:
                error_msg = f"Error getting future result for sample {sample_index}, answer {index}: {e}"
                print(error_msg, file=sys.stderr)
                # Store error message in the answer object
                line_data['answers'][index]['usefulness_analysis'] = error_msg
                ordered_bool_results[index] = False  # Default on error

    category = f"{yes_count}/3"
    line_data['category'] = category # Write category to the sample
    
    # 2. Calculate predicted ranking and add it to the item
    try:
        scores = [a['factuality_score'] for a in answers_data]
        predicted_ranking = get_score_ranking_order(scores)
        line_data['predicted_ranking'] = predicted_ranking
        
        # Log progress to console (now uses the ordered list)
        usefulness_str = ', '.join(['Y' if r else 'N' for r in ordered_bool_results])
        print(f"Sample {sample_index}: Usefulness=[{usefulness_str}] -> Category={category}.")
        
        return line_data # Return the modified sample
    
    except KeyError as e:
        print(f"Error processing rankings for sample {sample_index}: Missing key {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error processing rankings for sample {sample_index}: {e}", file=sys.stderr)
        return None

def process_sample_wrapper(args_tuple):
    """Wrapper function for pool.map to unpack arguments."""
    return process_sample(*args_tuple)

# --- User-Provided Evaluation Function ---

def evaluate_final_results_ranking(results: List[Dict]):
    """
    Calculates and prints evaluation metrics for the ranking scheme.
    The ground truth is a ranked list. Metrics used are P@1, Kendall's Tau, and NDCG.
    """
    kendall_tau_scores, ndcg_scores = [], []
    top_1_correct_count, valid_evaluation_count, invalid_predictions = 0, 0, 0
    total_items = len(results)
    
    for item in results:
        true_ranking = item.get("verify_result")
        pred_ranking = item.get("predicted_ranking")
        
        is_true_label_valid = isinstance(true_ranking, list) and true_ranking
        if not is_true_label_valid: continue

        is_pred_valid = isinstance(pred_ranking, list) and pred_ranking
        if not is_pred_valid or len(true_ranking) != len(pred_ranking):
            invalid_predictions += 1
            continue

        valid_evaluation_count += 1
        num_answers = len(true_ranking)

        if true_ranking[0] == pred_ranking[0]:
            top_1_correct_count += 1

        true_ranks = [0] * num_answers
        for rank, item_idx in enumerate(true_ranking):
            true_ranks[item_idx-1] = rank  # 存储每个 item_idx 对应的排名

        pred_ranks = [0] * num_answers
        for rank, item_idx in enumerate(pred_ranking):
            pred_ranks[item_idx-1] = rank  # 存储每个 item_idx 对应的排名

        tau, _ = kendalltau(true_ranks, pred_ranks)
        kendall_tau_scores.append(tau)

    if valid_evaluation_count == 0:
        logging.error("Evaluation failed. No valid items to evaluate.")
        return None

    precision_at_1 = top_1_correct_count / valid_evaluation_count if valid_evaluation_count else 0.0
    avg_kendall_tau = sum(kendall_tau_scores) / len(kendall_tau_scores) if kendall_tau_scores else 0.0
    # avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    invalid_ratio = invalid_predictions / total_items if total_items > 0 else 0.0

    metrics_dict = {
        "precision_at_1": round(precision_at_1, 4),
        "average_kendall_tau": round(avg_kendall_tau, 4),
        # "average_ndcg": round(avg_ndcg, 4),
        "invalid_prediction_ratio": round(invalid_ratio, 4),
    }

    print("\n--- Ranking Evaluation Results ---")
    for key, value in metrics_dict.items(): print(f"{key.replace('_', ' ').title()}: {value}")
    print("----------------------------------\n")
    return metrics_dict


# --- Main Execution ---

def main():
    # Define the input file path.
    # It will look for 'data.jsonl' in the same directory.
    dataset_name = "musique_new"
    model_name = "Qwen2.5-7B-Instruct"
    jsonl_file_path = f'/workspace/FactVeri-SFT/corpora/{dataset_name}/{dataset_name}_verification-{model_name}-retrieval-pointwise-sum_history-cls.json'
    output_jsonl_path = f'/workspace/FactVeri-SFT/corpora/{dataset_name}/{dataset_name}_verification-{model_name}-retrieval-pointwise-sum_history-cls-analysis.json' # New output file

    lines_to_process = []

    print(f"Processing '{jsonl_file_path}'...", file=sys.stderr)
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        lines_to_process = f.readlines()

    # This dict will store results: {"3/3": [item1, item2], ...}
    category_results = defaultdict(list)
    
    total_lines = len(lines_to_process)
    if total_lines == 0:
        print("No lines found to process.")
        return
        
    print(f"Found {total_lines} samples. Starting processing...")
    
    # Set number of processes
    try:
        # Use all available CPUs
        num_processes = os.cpu_count()
        print(f"Using {num_processes} processes for parallelization.")
    except NotImplementedError:
        num_processes = 4 # Fallback
        print(f"Could not detect CPU count, defaulting to {num_processes} processes.")

    # Prepare inputs for the pool
    inputs_for_pool = []
    for i, line in enumerate(lines_to_process):
        if not line.strip():
            continue
        try:
            line_data = json.loads(line)
            inputs_for_pool.append((line_data, i + 1))
        except json.JSONDecodeError:
            print(f"Skipping line {i+1}: Invalid JSON.", file=sys.stderr)
        except Exception as e:
            print(f"Skipping line {i+1}: Error preparing task {e}", file=sys.stderr)

    # Run multiprocessing pool
    all_results = []
    try:
        with Pool(processes=num_processes, initializer=init_worker, initargs=(counter, start_time)) as pool:
            all_results = pool.map(process_sample_wrapper, inputs_for_pool)
    except Exception as e:
        print(f"An error occurred during multiprocessing: {e}", file=sys.stderr)

    # Filter out None results (from errors) and populate category_results
    valid_results = []
    for item in all_results:
        if item is not None:
            valid_results.append(item)
            try:
                category_results[item['category']].append(item)
            except KeyError:
                print(f"Warning: Processed item missing 'category' key: {item}", file=sys.stderr)


    # Write results to the output jsonl file
    try:
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for item in valid_results:
                f.write(json.dumps(item) + '\n')
        print(f"\nSuccessfully saved {len(valid_results)} analyzed samples to {output_jsonl_path}")
    except IOError as e:
        print(f"\nError writing to output file {output_jsonl_path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"\nAn unknown error occurred while writing output file: {e}", file=sys.stderr)


    end_time_total = time.time()
    print("\n--- Processing Complete ---")
    print(f"Total time: {end_time_total - start_time:.2f} seconds")
    print(f"Total samples processed: {total_lines}")
    print(f"Total API calls made: {counter.value}")

    print("\n--- Results by Category ---")
    categories_to_report = ["3/3", "2/3", "1/3", "0/3"]
    
    for category in categories_to_report:
        results_list = category_results[category]
        print(f"\n--- Category {category} (Total Samples: {len(results_list)}) ---")
        
        if results_list:
            evaluate_final_results_ranking(results_list)
        else:
            print("No samples in this category.")

if __name__ == "__main__":
    main()