export CUDA_VISIBLE_DEVICES=7

DATA=nq_hotpotqa_train

MODELS=(
    "Llama-3.1-8B-Instruct" 
    "Qwen2-7B-Instruct" 
    "Qwen2.5-7B-Instruct" 
    "Llama-3.2-3B-Instruct" 
    "Meta-Llama-3-8B-Instruct"
    "gemma-2-2b-it"
    "gemma-3-1b-it"
    "Phi-3-mini-4k-instruct"
    "Phi-4-mini-instruct"
    "Qwen2.5-3B-Instruct"
    "Qwen2.5-14B-Instruct"
)

# if [ ! -f ./corpora/$DATA/$DATA\_generation.jsonl ]; then
#     echo "./corpora/$DATA/$DATA\_generation.jsonl does not exist. Constructing it."
#     cp ./corpora/$DATA/$DATA\_question.jsonl ./corpora/$DATA/$DATA\_generation.jsonl
# fi

# for MODEL in "${MODELS[@]}"
# do
#     python inference/generate_answer.py \
#         --model-type vllm \
#         --model-path /workspace/HFModels/$MODEL \
#         --input-file ./corpora/$DATA/$DATA\_generation.jsonl \
#         --output-file ./corpora/$DATA/$DATA\_generation.jsonl \
#         --phase "generation"
# done

MODEL="/workspace/HFModels/Qwen3-14B"
python inference/generate_answer.py \
    --model-type vllm \
    --model-path $MODEL \
    --input-file ./corpora/$DATA/$DATA\_generation.jsonl \
    --output-file ./corpora/$DATA/$DATA\_evaluation.jsonl \
    --phase "evaluation" \
    --negative-num 1 \
    --multi-process "True"

# python inference/generate_answer.py \
#     --model-type vllm \
#     --model-path /workspace/HFModels/$MODEL \
#     --input-file ./corpora/$DATA/$DATA\_evaluation.jsonl \
#     --output-file ./corpora/$DATA/$DATA\_selection.jsonl \
#     --phase "selection" \
#     --negative-num 1

# MODEL="Qwen2.5-14B-Instruct"
# python inference/generate_answer.py \
#     --model-type vllm \
#     --model-path /workspace/HFModels/$MODEL \
#     --input-file ./corpora/$DATA/$DATA\_selection.jsonl \
#     --output-file ./corpora/$DATA/$DATA\_verification.jsonl \
#     --phase "verification"