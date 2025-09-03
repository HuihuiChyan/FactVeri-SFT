import json
import random
from datasets import load_dataset

# Load the dataset
arc_dataset = load_dataset("allenai/qasc")
test_data = arc_dataset['validation']

# --- You can adjust this number as needed ---
# Process all entries by default
data_entries = [data_entry for data_entry in test_data][:300]

# Define the output filename
output_filename = "qasc_selection.jsonl"

print(f"正在将问题和选项写入 {output_filename} (Writing questions and choices to {output_filename})...")

# Open the output file
with open(output_filename, 'w', encoding='utf-8') as fout:
    # Loop through each entry in the dataset
    for data_entry in data_entries:

        all_answers = data_entry["choices"]["text"]
        correct_answer_idx = data_entry["choices"]["label"].index(data_entry["answerKey"])
        correct_answer = all_answers[correct_answer_idx]
        incorrect_answers = all_answers
        incorrect_answers.remove(correct_answer)

        all_answers = [correct_answer] + incorrect_answers[:3]

        random.shuffle(all_answers)

        verify_result = all_answers.index(correct_answer)
        
        all_answers = [{"answer": answer} for answer in all_answers]

        # Create a dictionary with only the desired fields
        output_data = {
            "question": data_entry['question'],
            "reference": correct_answer,
            "answers": all_answers,
            "verify_result": verify_result # Also keeping the answerKey for reference
        }
        
        # Write the JSON object as a single line to the file
        fout.write(json.dumps(output_data, ensure_ascii=False) + '\n')

print(f"所有操作完成！文件已保存到 {output_filename} (All operations complete! File saved to {output_filename}).")