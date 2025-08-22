export CUDA_VISIBLE_DEVICES=5

model_path=/workspace/HFModels/
model_name=Qwen2.5-7B-Instruct
mode=local_retrieval
scheme=pointwise
dataset_path=/workspace/FactVeri-data/hotpotqa
dataset_name_without_ext=hotpotqa_$scheme
python -u inference/infer_batch_sglang.py \
    --model_path $model_path/$model_name \
    --input_file $dataset_path/$dataset_name_without_ext.jsonl \
    --output_file ./results/$dataset_name_without_ext-$model_name-$mode-$scheme.jsonl \
    --mode $mode \
    --scheme $scheme