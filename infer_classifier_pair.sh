export CUDA_VISIBLE_DEVICES=7

dataset_path=/workspace/FactVeri-data/bamboogle
dataset_name_without_ext=bamboogle_pointwise
model_path=/workspace/HFModels
model_name=Qwen2.5-0.5B-Instruct-RM-no-trace/final_model

python -u inference/infer_classifier_pair.py \
    --model_path $model_path/$model_name \
    --input_file $dataset_path/$dataset_name_without_ext.jsonl \
    --output_file ./results/$dataset_name_without_ext-cls.jsonl \