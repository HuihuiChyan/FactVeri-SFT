import json
from datasets import load_dataset

# Load the dataset
arc_dataset = load_dataset("allenai/ai2_arc", 'ARC-Challenge')
test_data = arc_dataset['test']

# --- You can adjust this number as needed ---
# Process all entries by default
data_entries = [data_entry for data_entry in test_data][:300]

# Define the output filename
output_filename = "ai2_arc_selection.jsonl"

print(f"正在将问题和选项写入 {output_filename} (Writing questions and choices to {output_filename})...")

# Open the output file
with open(output_filename, 'w', encoding='utf-8') as fout:
    # Loop through each entry in the dataset
    for data_entry in data_entries:

        choices = [{"answer": answer} for answer in data_entry["choices"]["text"]]

        key2idx = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
        
        # Create a dictionary with only the desired fields
        output_data = {
            "question": data_entry['question'],
            "reference": choices[key2idx[data_entry['answerKey']]],
            "answers": choices,
            "verify_result": key2idx[data_entry['answerKey']] # Also keeping the answerKey for reference
        }
        
        # Write the JSON object as a single line to the file
        fout.write(json.dumps(output_data, ensure_ascii=False) + '\n')

print(f"所有操作完成！文件已保存到 {output_filename} (All operations complete! File saved to {output_filename}).")