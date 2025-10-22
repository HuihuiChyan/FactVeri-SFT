export CUDA_VISIBLE_DEVICES=7

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export SERPER_KEY_PRIVATE=95cc94f4818a2ffbc6b80a3c935d5729a24a087f

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

if [ ! -f ./corpora/$DATA/$DATA\_generation.jsonl ]; then
    echo "./corpora/$DATA/$DATA\_generation.jsonl does not exist. Constructing it."
    cp ./corpora/$DATA/$DATA\_question.jsonl ./corpora/$DATA/$DATA\_generation.jsonl
fi

# for MODEL in "${MODELS[@]}"
# do
#     python inference/generate_answer.py \
#         --model-type vllm \
#         --model-path /workspace/HFModels/$MODEL \
#         --input-file ./corpora/$DATA/$DATA\_generation.jsonl \
#         --output-file ./corpora/$DATA/$DATA\_generation.jsonl \
#         --phase "generation"
# done

# MODEL="/workspace/HFModels/Qwen3-8B"
# python inference/generate_answer.py \
#     --model-type vllm \
#     --model-path $MODEL \
#     --input-file ./corpora/$DATA/$DATA\_generation.jsonl \
#     --output-file ./corpora/$DATA/$DATA\_evaluation.jsonl \
#     --phase "evaluation" \
#     --multi-process "True"

python inference/generate_answer.py \
    --model-type vllm \
    --model-path /workspace/HFModels/$MODEL \
    --input-file ./corpora/$DATA/$DATA\_evaluation.jsonl \
    --output-file ./corpora/$DATA/$DATA\_selection.jsonl \
    --phase "selection"

MODEL="gpt-4o"
python inference/generate_answer.py \
    --model-type api \
    --model-path $MODEL \
    --input-file ./corpora/$DATA/$DATA\_selection.jsonl \
    --output-file ./corpora/$DATA/$DATA\_verification.jsonl \
    --phase "verification"