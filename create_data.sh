export CUDA_VISIBLE_DEVICES=0

DATA=science_qa

python construction/generate_answer.py \
    --model-path "gpt-4o" \
    --input-file ./corpora/$DATA/$DATA\_question.jsonl \
    --output-file ./corpora/$DATA/$DATA\_combination.jsonl \
    --phase "combination"

# python construction/generate_answer.py \
#     --model-path "gpt-4o" \
#     --input-file ./corpora/$DATA/$DATA\_combination.jsonl \
#     --output-file ./corpora/$DATA/$DATA\_verification.jsonl \
#     --phase "verification"