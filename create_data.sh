export CUDA_VISIBLE_DEVICES=0,5,6,7

DATA=bamboogle

MODELS=("Llama-3.1-8B-Instruct" "Qwen2-7B-Instruct" "Qwen2.5-7B-Instruct" "Llama-3.2-3B-Instruct" "Meta-Llama-3-8B-Instruct")

if [ ! -f ./corpora/$DATA/$DATA\_generation.jsonl ]; then
    echo "./corpora/$DATA/$DATA\_generation.jsonl does not exist. Constructing it."
    cp ./corpora/$DATA/$DATA\_question.jsonl ./corpora/$DATA/$DATA\_generation.jsonl
fi

# for MODEL in "${MODELS[@]}"
# do
#     python construction/generate_answer.py \
#         --model-type vllm \
#         --model-mode correct \
#         --model-path /workspace/HFModels/$MODEL \
#         --input-file ./corpora/$DATA/$DATA\_generation.jsonl \
#         --output-file ./corpora/$DATA/$DATA\_generation.jsonl \
#         --phase "generation"
#     python construction/generate_answer.py \
#         --model-type vllm \
#         --model-mode incorrect \
#         --model-path /workspace/HFModels/$MODEL \
#         --input-file ./corpora/$DATA/$DATA\_generation.jsonl \
#         --output-file ./corpora/$DATA/$DATA\_generation.jsonl \
#         --phase "generation"
# done

# MODEL="Qwen2.5-14B-Instruct"
# python construction/generate_answer.py \
#     --model-type vllm \
#     --model-path /workspace/HFModels/$MODEL \
#     --input-file ./corpora/$DATA/$DATA\_generation.jsonl \
#     --output-file ./corpora/$DATA/$DATA\_evaluation.jsonl \
#     --phase "evaluation"

# python construction/generate_answer.py \
#     --model-type vllm \
#     --model-path /workspace/HFModels/$MODEL \
#     --input-file ./corpora/$DATA/$DATA\_evaluation.jsonl \
#     --output-file ./corpora/$DATA/$DATA\_selection.jsonl \
#     --phase "selection"

MODEL="Qwen2.5-14B-Instruct"
python construction/generate_answer.py \
    --model-type vllm \
    --model-path /workspace/HFModels/$MODEL \
    --input-file ./corpora/$DATA/$DATA\_selection.jsonl \
    --output-file ./corpora/$DATA/$DATA\_verification.jsonl \
    --phase "verification"