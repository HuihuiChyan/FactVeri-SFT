export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export SERPER_KEY_PRIVATE=95cc94f4818a2ffbc6b80a3c935d5729a24a087f

model_path=/workspace/HFModels/
model_name=Qwen2.5-7B-Instruct
mode=retrieval # choose between retrieval and direct_gen
scheme=pointwise
dataset_path=/workspace/FactVeri-SFT/corpora/nq_hotpotqa_train
dataset_name_without_ext=nq_hotpotqa_train_verification
# python -u inference/infer_batch_sglang.py \
#     --model_path $model_path/$model_name \
#     --input_file $dataset_path/$dataset_name_without_ext.jsonl \
#     --output_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme.json \
#     --mode $mode \
#     --scheme $scheme \
#     --disable_cache_for_serper

cls_input=full_history # choose between full_history, sum_history and direct_input

# python -u inference/infer_sum_pointwise.py \
#     --model_path $model_path/$model_name \
#     --input_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme.json \
#     --output_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme-sum.json \

# python -u inference/process_cls_input.py \
#     --input-file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme-sum-convert.json \
#     --cls-input $cls_input \
#     --output-file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme-$cls_input.json \
#     --tokenizer-path $model_path/$model_name

accelerate launch \
    --num_processes 8 \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file examples/deepspeed/ds_z2_config.json \
    inference/train_btcls.py \
    --model_name $model_path/$model_name \
    --data_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme-$cls_input.json \
    --output_dir $model_path/$model_name-RM-$mode-$cls_input-2e-4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 \
    --max_length 1024 \
    --logging_steps 10 \
    --save_strategy epoch \
    --gradient_checkpointing False \
    --use_lora \
    --bf16