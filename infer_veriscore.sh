export CUDA_VISIBLE_DEVICES=1

export SERPER_KEY_PRIVATE=95cc94f4818a2ffbc6b80a3c935d5729a24a087f
export OPENAI_API_KEY=sk-XI5BH5GCT4jyfoYklI5BklGiJjX6C4Tw9M6UeAYVVu2xI2V5
export OPENAI_BASE_URL=https://yunwu.ai/v1

model_name=gpt-4o
dataset=triviaqa_new
dataset_path=/root/autodl-tmp/FactVeri-SFT/corpora/$dataset
dataset_name_without_ext=$dataset\_verification

python ablation/run_veriscore.py \
    --model_name $model_name \
    --input_file $dataset_path/$dataset_name_without_ext.jsonl \
    --output_file $dataset_path/$dataset_name_without_ext-$model_name-veriscore.jsonl

python -u ablation/compare_scoring.py \
    $dataset_path/$dataset_name_without_ext-$model_name-veriscore.jsonl
