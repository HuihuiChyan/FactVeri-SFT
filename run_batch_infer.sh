export CUDA_VISIBLE_DEVICES=0,1,2

dataset_path=/workspace/FactVeri-R1/fact_checking_dataset
dataset_name_without_ext=bamboogle_judged #hotpotqa_subset_200
model_path=/workspace/HFModels
model_name=Qwen2.5-7B-Instruct
mode=local_retrieval
python infer_batch.py \
    --model_path $model_path/$model_name \
    --input_file $dataset_path/$dataset_name_without_ext.jsonl  \
    --output_file ./results/$dataset_name_without_ext-$model_name.jsonl \
    --log_file ./log/run_$dataset_name_without_ext-$model_name-$mode.log \
    --mode $mode

# python infer_batch_qa.py \
#     --model_path $model_path/$model_name \
#     --input_file $dataset_path/$dataset_name_without_ext.jsonl  \
#     --output_file ./results/$dataset_name_without_ext-$model_name.jsonl \
#     --log_file ./log/run_$dataset_name_without_ext-$model_name.log \
#     --mode local_retrieval