export CUDA_VISIBLE_DEVICES=4

dataset_path=/workspace/FactVeri-data/bamboogle
dataset_name_without_ext=bamboogle_test
model_path=/workspace/HFModels/
model_name=Qwen2.5-7B-Instruct
mode=local_retrieval
python -u infer_batch_sglang.py \
    --model_path $model_path/$model_name \
    --input_file $dataset_path/$dataset_name_without_ext.jsonl \
    --output_file ./results/$dataset_name_without_ext-$model_name-$mode.jsonl \
    --mode $mode

# python infer_batch_qa.py \
#     --model_path $model_path/$model_name \
#     --input_file $dataset_path/$dataset_name_without_ext.jsonl  \
#     --output_file ./results/$dataset_name_without_ext-$model_name.jsonl \
#     --log_file ./log/run_$dataset_name_without_ext-$model_name.log \
#     --mode local_retrieval