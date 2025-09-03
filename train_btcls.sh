export CUDA_VISIBLE_DEVICES=4,5,6,7

model_name=Qwen2.5-0.5B-Instruct
cls_input=trace
model_path=/workspace/HFModels/${model_name}
output_prefix=/workspace/FactVeri-SFT/results/nq_hotpot_train_head_selection-Qwen3-4B-pointwise-local_retrieval

# # Run the conversion script on the final merged file
# python -u inference/process_cls_input.py \
#     --input-file ${output_prefix}.json \
#     --cls-input $cls_input \
#     --output-file ${output_prefix}-${cls_input}.json \
#     --tokenizer-path ${model_path}

accelerate launch --num_processes 4 --deepspeed_config_file examples/deepspeed/ds_z2_config.json inference/train_btcls.py \
    --model_name ${model_path} \
    --data_file ${output_prefix}-${cls_input}.json \
    --output_dir  /workspace/HFModels/${model_name}-RM-${cls_input} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-6 \
    --max_length 1024 \
    --logging_steps 10 \
    --save_strategy no \
    --gradient_checkpointing True \
    --bf16 True  # 直接在这里启用混合精度！ 