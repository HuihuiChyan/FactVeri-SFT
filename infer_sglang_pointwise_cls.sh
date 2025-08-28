export CUDA_VISIBLE_DEVICES=0

model_path=/workspace/HFModels/
model_name=Qwen2.5-7B-Instruct
mode=local_retrieval # choose between local_retrieval and direct_gen
scheme=pointwise
dataset_path=/workspace/FactVeri-SFT/corpora/hotpotqa
dataset_name_without_ext=hotpotqa_testset_pointwise
# python -u inference/infer_batch_sglang.py \
#     --model_path $model_path/$model_name \
#     --input_file $dataset_path/$dataset_name_without_ext.jsonl \
#     --output_file ./results/$dataset_name_without_ext-$model_name-$scheme.json \
#     --mode $mode \
#     --scheme $scheme

# python -u inference/convert_pointwise_to_pairwise_trace.py \
#     --input_file ./results/$dataset_name_without_ext-$model_name-$scheme.json \
#     --output_file ./results/$dataset_name_without_ext-$model_name-$scheme-trace.json

infer_model_path=/workspace/HFModels
infer_model_name=Qwen2.5-0.5B-Instruct-RM-reformat-trace/final_model

python -u inference/infer_classifier_pair.py \
    --model_path $infer_model_path/$infer_model_name \
    --input_file ./results/$dataset_name_without_ext-$model_name-$scheme-trace.json \
    --output_file ./results/$dataset_name_without_ext-$model_name-$scheme-trace-cls.json \