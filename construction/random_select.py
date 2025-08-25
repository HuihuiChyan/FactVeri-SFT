import json
import random
INPUT = "/workspace/FactVeri-data/musique/musique_filtering_full.jsonl"
OUTPUT = "/workspace/FactVeri-data/musique/musique_filtering.jsonl"

random.seed(42)

with open(INPUT, "r", encoding="utf-8") as fin,\
open(OUTPUT, "w", encoding="utf-8") as fout:
    lines = [json.loads(line.strip()) for line in fin.readlines()]
    new_lines = random.sample(lines, k=500)
    for line in new_lines:
        fout.write(json.dumps(line) + "\n")