export CUDA_VISIBLE_DEVICES=7

dataset_path=/workspace/FactVeri-data/bamboogle
dataset_name_without_ext=bamboogle_test
model_path=/workspace/HFModels
model_name=Qwen2.5-3B-Instruct-nq_hotpot_train_head_test_local_retrieval_no_trace
mode=local_retrieval
python -u infer_classifier.py \
    --model_path $model_path/$model_name \
    --input_file $dataset_path/$dataset_name_without_ext.jsonl \
    --output_file ./results/$dataset_name_without_ext-cls.jsonl \