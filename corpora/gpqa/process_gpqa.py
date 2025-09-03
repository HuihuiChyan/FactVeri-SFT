import json
import random
from datasets import load_dataset

# Load the dataset
arc_dataset = load_dataset("Idavidrein/gpqa", 'gpqa_main')
test_data = arc_dataset['train']

# --- You can adjust this number as needed ---
# Process all entries by default
data_entries = [data_entry for data_entry in test_data][:300]

# Define the output filename
output_filename = "gpqa_selection.jsonl"

print(f"正在将问题和选项写入 {output_filename} (Writing questions and choices to {output_filename})...")

# Open the output file
with open(output_filename, 'w', encoding='utf-8') as fout:
    # Loop through each entry in the dataset
    for data_entry in data_entries:

        correct_answer = data_entry["Correct Answer"]
        incorrect_answers = [data_entry["Incorrect Answer 1"], data_entry["Incorrect Answer 2"], data_entry["Incorrect Answer 3"]]

        all_answers = [correct_answer] + incorrect_answers

        random.shuffle(all_answers)

        verify_result = all_answers.index(correct_answer)
        
        all_answers = [{"answer": answer} for answer in all_answers]

        # Create a dictionary with only the desired fields
        output_data = {
            "question": data_entry['Question'],
            "reference": correct_answer,
            "answers": all_answers,
            "verify_result": verify_result # Also keeping the answerKey for reference
        }
        
        # Write the JSON object as a single line to the file
        fout.write(json.dumps(output_data, ensure_ascii=False) + '\n')

print(f"所有操作完成！文件已保存到 {output_filename} (All operations complete! File saved to {output_filename}).")