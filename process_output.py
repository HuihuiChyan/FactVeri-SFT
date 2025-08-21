import json

INPUT = "/workspace/Search-R1/results/nq_hotpot_train_head_test_local_retrieval_merged.jsonl"
OUTPUT_FULL = "/workspace/LLaMA-Factory/data/nq_hotpot_train_head_test_local_retrieval_full_trace.json"
OUTPUT_NO = "/workspace/LLaMA-Factory/data/nq_hotpot_train_head_test_local_retrieval_no_trace.json"

def extract_dialogue(text):
    """
    Extracts dialogue turns and roles from the provided text.

    Args:
        text (str): The input text containing the conversation.

    Returns:
        list: A list of dictionaries, where each dictionary represents a dialogue turn
              and contains 'role' and 'content' keys.
    """
    import re

    # Pattern to find dialogue turns and their roles
    # The pattern looks for '<|im_start|>role\n' and then captures the content until '<|im_end|>'
    pattern = r'<\|im_start\|>(system|user|assistant)\n(.*?)(?=<\|im_end\|>)'
    
    dialogue_list = []
    
    # Use re.DOTALL to make '.' match newlines
    matches = re.findall(pattern, text, re.DOTALL)
    
    for role, content in matches:
        # Clean up the content by stripping leading/trailing whitespace
        content = content.strip()
        
        # Create a dictionary for each turn and add it to the list
        dialogue_list.append({
            "role": role,
            "content": content
        })
        
    return dialogue_list

def remove_final_verdict(text):
    verdict_start = text.rfind("Therefore, the final verdict is: <answer>")
    if verdict_start != -1:
        return text[:verdict_start].strip()
    return text   

with open(INPUT, "r", encoding="utf-8") as fin,\
open(OUTPUT_FULL, "w", encoding="utf-8") as fout_full,\
open(OUTPUT_NO, "w", encoding="utf-8") as fout_no:
    lines = [json.loads(line.strip()) for line in fin.readlines()]

    output_lines_full = []
    output_lines_no = []

    for line in lines:
        dialogue = extract_dialogue(line["model_output_trace"])
        for i, turn in enumerate(dialogue):
            if i == 0:
                assert turn["role"] == "system"
            if i != 0 and i % 2 == 1:
                assert turn["role"] == "user"
                turn["from"] = "human"
                turn["value"] = turn["content"]
                del turn["role"]
                del turn["content"]
            if i != 0 and i % 2 == 0:
                assert turn["role"] == "assistant"
                turn["from"] = "gpt"
                turn["value"] = turn["content"]
                del turn["role"]
                del turn["content"]
            
            if i == len(dialogue)-1:
                turn["value"] = remove_final_verdict(turn["value"])

        if line["final_verdict"].lower() != line["label"]:
            continue
        new_line = {
            "conversations": dialogue[1:],
            "system": dialogue[0]["content"],
            "verify_result": int(line["label"] == "supported"),
        }
        output_lines_full.append(new_line)

        new_no_line = {
            "conversations": [
                {
                    "from": "human",
                    "value": line["question"],
                },
                {
                    "from": "gpt",
                    "value": line["response"],                    
                }
            ],
            "verify_result": int(line["label"] == "supported"),
        }
        output_lines_no.append(new_no_line)

    print(f"Totally {len(output_lines_no)} lines.")
    json.dump(output_lines_full, fout_full, indent=4)
    json.dump(output_lines_no, fout_no, indent=4)