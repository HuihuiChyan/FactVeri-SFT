export CUDA_VISIBLE_DEVICES=2

DATA=2wiki

# python construction/generate_answer.py \
#     --model-path "gpt-4o" \
#     --temperature 0.0 \
#     --input-file $DATA/$DATA\_question.jsonl \
#     --output-file $DATA/$DATA\_filtering.jsonl \
#     --phase "filtering"

python construction/generate_answer.py \
    --model-path "/workspace/HFModels/Qwen2.5-3B-Instruct" \
    --input-file $DATA/$DATA\_filtering.jsonl \
    --output-file $DATA/$DATA\_generation.jsonl \
    --model-type "vllm" \
    --phase "generation"

# python generate_answer.py \
#     --model-path "/workspace/HFModels/Qwen2.5-7B-Instruct" \
#     --input-file $DATA/$DATA\_generation.jsonl \
#     --output-file $DATA/$DATA\_generation.jsonl \
#     --model-type "vllm" \
#     --phase "generation"

# python generate_answer.py \
#     --model-path "/workspace/HFModels/Qwen2.5-14B-Instruct" \
#     --input-file $DATA/$DATA\_generation.jsonl \
#     --output-file $DATA/$DATA\_generation.jsonl \
#     --model-type "vllm" \
#     --phase "generation"

# python generate_answer.py \
#     --model-path "/workspace/HFModels/Meta-Llama-3-8B-Instruct" \
#     --input-file $DATA/$DATA\_generation.jsonl \
#     --output-file $DATA/$DATA\_generation.jsonl \
#     --model-type "vllm" \
#     --phase "generation"

# python generate_answer.py \
#     --model-path "/workspace/HFModels/Mistral-7B-Instruct-v0.3" \
#     --input-file $DATA/$DATA\_generation.jsonl \
#     --output-file $DATA/$DATA\_generation.jsonl \
#     --model-type "vllm" \
#     --phase "generation"

# python generate_answer.py \
#     --model-path "/workspace/HFModels/Llama-3.1-8B-Instruct" \
#     --input-file $DATA/$DATA\_generation.jsonl \
#     --output-file $DATA/$DATA\_generation.jsonl \
#     --model-type "vllm" \
#     --phase "generation"

python generate_answer.py \
    --model-path "gpt-4o" \
    --input-file $DATA/$DATA\_generation.jsonl \
    --output-file $DATA/$DATA\_verification.jsonl \
    --phase "verification"

python generate_answer.py \
    --model-path "gpt-4o" \
    --input-file $DATA/$DATA\_verification.jsonl \
    --output-file $DATA/$DATA\_combination.jsonl \
    --phase "combination"