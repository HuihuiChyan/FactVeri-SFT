export CUDA_VISIBLE_DEVICES=6

model_path=/workspace/HFModels/
model_name=Qwen2.5-7B-Instruct
mode=local_retrieval # choose between local_retrieval and direct_gen
scheme=pointwise
dataset=triviaqa
dataset_path=/workspace/FactVeri-SFT/corpora/$dataset
dataset_name_without_ext=$dataset\_verification
python -u inference/infer_batch_sglang.py \
    --model_path $model_path/$model_name \
    --input_file $dataset_path/$dataset_name_without_ext.jsonl \
    --output_file ./results/$dataset_name_without_ext-$model_name-$mode-$scheme.json \
    --mode $mode \
    --scheme $scheme \
    --disable_thinking

# cls_input_mode=trace
# infer_model_path=/workspace/HFModels
# infer_model_name=Qwen2.5-3B-Instruct-RM-$cls_input_mode/final_model

# python -u inference/infer_classifier.py \
#     --model_path $infer_model_path/$infer_model_name \
#     --cls-input $cls_input_mode \
#     --input_file ./results/$dataset_name_without_ext-$model_name-$scheme.json \
#     --output_file ./results/$dataset_name_without_ext-$model_name-$scheme-cls.json \