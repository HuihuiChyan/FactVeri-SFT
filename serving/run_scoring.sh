#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export SERPER_KEY_PRIVATE=95cc94f4818a2ffbc6b80a3c935d5729a24a087f

model_path=/root/autodl-tmp/HFModels/
model_name=Qwen2.5-7B-Instruct
mode=retrieval  # choose between retrieval and direct_score
dataset=triviaqa_new
dataset_path=/root/autodl-tmp/FactVeri-SFT/corpora/$dataset
dataset_name_without_ext=$dataset\_verification

# SGLang 服务配置
sglang_url=http://localhost:30000

# 运行推理
# python -u infer_scoring.py \
#     --sglang_url $sglang_url \
#     --tokenizer_path $model_path/$model_name \
#     --input_file $dataset_path/$dataset_name_without_ext.jsonl \
#     --output_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-scoring.json \
#     --mode $mode \
#     --max_token 2048 \
#     --temperature 0.0 \
#     --num_threads 10 \
#     --no-use-serper-cache

# python -u evaluate_results.py \
#     --input_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-scoring.json

python -u infer_sum_pointwise.py \
    --tokenizer_path $model_path/$model_name \
    --input_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-scoring.json \
    --output_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-sum.json \
    --sglang_url $sglang_url \
    --num_threads 10

python -u evaluate_results.py \
    --input_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-sum.json