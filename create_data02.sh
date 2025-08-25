export CUDA_VISIBLE_DEVICES=6

DATA=hotpotqa

# python construction/generate_answer.py \
#     --model-path "gpt-4o" \
#     --temperature 0.0 \
#     --input-file $DATA/$DATA\_question.jsonl \
#     --output-file $DATA/$DATA\_filtering.jsonl \
#     --phase "filtering"

cp ./corpora/$DATA/$DATA\_filtering.jsonl ./corpora/$DATA/$DATA\_generation.jsonl

MODELS=(
    "Llama-3.1-8B-Instruct"
    "Meta-Llama-3-8B-Instruct"
    "Mistral-7B-Instruct-v0.3"
    "Qwen2-7B-Instruct"
    "Qwen2.5-14B-Instruct"
    "Qwen2.5-7B-Instruct"
)

# for MODEL in "${MODELS[@]}"; do
#     python construction/generate_answer.py \
#         --model-path /workspace/HFModels/$MODEL \
#         --input-file ./corpora/$DATA/$DATA\_generation.jsonl \
#         --output-file ./corpora/$DATA/$DATA\_generation.jsonl \
#         --model-type "vllm" \
#         --phase "generation"
# done

# python construction/generate_answer.py \
#     --model-path "gpt-4o" \
#     --input-file ./corpora/$DATA/$DATA\_generation.jsonl \
#     --output-file ./corpora/$DATA/$DATA\_scoring.jsonl \
#     --phase "scoring"

python construction/generate_answer.py \
    --model-path "gpt-4o" \
    --input-file ./corpora/$DATA/$DATA\_scoring.jsonl \
    --output-file ./corpora/$DATA/$DATA\_selection.jsonl \
    --phase "selection"

python construction/generate_answer.py \
    --model-path "gpt-4o" \
    --input-file ./corpora/$DATA/$DATA\_selection.jsonl \
    --output-file ./corpora/$DATA/$DATA\_verification.jsonl \
    --phase "verification"