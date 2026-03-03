export CUDA_VISIBLE_DEVICES=0

export SERPER_KEY_PRIVATE=95cc94f4818a2ffbc6b80a3c935d5729a24a087f

model_path=/root/autodl-tmp/HFModels/
model_name=Qwen2.5-7B-Instruct
mode=retrieval # choose between retrieval and direct_gen
scheme=scoring
dataset=hotpotqa_new
dataset_path=/root/autodl-tmp/FactVeri-SFT/corpora/$dataset
dataset_name_without_ext=$dataset\_verification
python -u src/infer_verify.py \
    --model_path $model_path/$model_name \
    --input_file $dataset_path/$dataset_name_without_ext.jsonl \
    --output_file $dataset_path/$dataset_name_without_ext-verify.json \
    --mode $mode \
    --scheme $scheme

python -u ablation/compare_scoring.py \
    $dataset_path/$dataset_name_without_ext-verify.json