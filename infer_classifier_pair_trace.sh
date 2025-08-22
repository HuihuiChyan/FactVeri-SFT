export CUDA_VISIBLE_DEVICES=6

dataset_path=/workspace/FactVeri-SFT/results/
dataset_name_without_ext=hotpotqa_pointwise-Qwen2.5-7B-Instruct-local_retrieval-pointwise_doc
model_path=/workspace/HFModels
model_name=Qwen2.5-0.5B-Instruct-RM-full-trace/final_model

python -u inference/infer_classifier_pair.py \
    --model_path $model_path/$model_name \
    --input_file $dataset_path/$dataset_name_without_ext.jsonl \
    --output_file ./results/$dataset_name_without_ext-cls.jsonl \