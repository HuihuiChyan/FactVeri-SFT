export CUDA_VISIBLE_DEVICES=5

model_path=/workspace/HFModels/
model_name=Qwen2.5-7B-Instruct
mode=local_retrieval # choose between local_retrieval and direct_gen
scheme=pointwise
dataset=nq_test
dataset_path=/workspace/FactVeri-SFT/corpora/$dataset
dataset_name_without_ext=$dataset\_selection
python -u inference/infer_batch_sglang.py \
    --model_path $model_path/$model_name \
    --input_file $dataset_path/$dataset_name_without_ext.jsonl \
    --output_file ./results/$dataset_name_without_ext-$model_name-$scheme.json \
    --mode $mode \
    --scheme $scheme

cls_input_mode=trace
infer_model_path=/workspace/HFModels
infer_model_name=Qwen2.5-3B-Instruct-RM-$cls_input_mode/final_model

python -u inference/infer_classifier.py \
    --model_path $infer_model_path/$infer_model_name \
    --cls-input $cls_input_mode \
    --lora_path /workspace/HFModels/Qwen2.5-0.5B-Instruct-RM-trace-LoRA/lora_weights \
    --input_file ./results/$dataset_name_without_ext-$model_name-$scheme-$search_api.json \
    --output_file ./results/$dataset_name_without_ext-$model_name-$scheme-$search_api-cls.json
