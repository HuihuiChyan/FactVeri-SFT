import json

input_file_path = '/workspace/FactVeri-SFT/results/nq_hotpot_train_head_selection-Qwen2.5-7B-Instruct-pointwise-local_retrieval-train.jsonl'
output_file_path = '/workspace/FactVeri-SFT/results/nq_hotpot_train_head_selection-Qwen2.5-7B-Instruct-pointwise-local_retrieval-train.json'

with open(input_file_path, 'r', encoding='utf-8') as infile, \
     open(output_file_path, 'w', encoding='utf-8') as outfile:
    
    for line in infile:
        try:
            # Load the JSON object from the line
            data = json.loads(line)
            
            # Check if the 'reference' field exists and is a string
            if 'reference' in data and isinstance(data['reference'], str):
                # If it's a string, wrap it in a list
                data['reference'] = [data['reference']]
            
            data["verify_result"] = 0
            
            # Write the corrected JSON object back to the new file
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

        except json.JSONDecodeError:
            print(f"Skipping malformed line: {line.strip()}")

print(f"Cleaned data saved to: {output_file_path}")