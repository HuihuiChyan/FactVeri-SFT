import json
INPUT = "/workspace/FactVeri-data/bamboogle/bamboogle_combination.jsonl"

DEFAULT_USER_CONTENT_PREFIX = "You are an expert fact-checking assistant. Your goal is to determine whether the following claim is real or not. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <tool_call> query </tool_call> and it will return the top searched results between <tool_response> and </tool_response>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer 'Real' or 'Not Real' inside <answer> and </answer>, without detailed illustrations. For example, <answer> Real </answer>. Now, begin your work for the following claim: {{claim}}"

with open(INPUT, "r", encoding="utf-8") as fin:
    lines = [json.loads(line.strip()) for line in fin.readlines()]
    new_lines = []
    for line in lines:
        new_line = {
            "conversations": [
                {
                    "from": "human",
                    "value": DEFAULT_USER_CONTENT_PREFIX.format(claim=line["positive_statement"])
                },
                {
                    "from": "human",
                    "value": DEFAULT_USER_CONTENT_PREFIX.format(claim=line["negative_statement"])
                },                
            ]
        }