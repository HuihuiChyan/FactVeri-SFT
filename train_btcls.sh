export CUDA_VISIBLE_DEVICES=4,6

output_prefix=/workspace/FactVeri-SFT/results/nq_hotpot_train_head_selection-Qwen2.5-7B-Instruct-pointwise-local_retrieval

# Run the conversion script on the final merged file
cls_input="facts"
python -u inference/process_cls_input.py \
    --input-file ${output_prefix}.json \
    --cls-input $cls_input \
    --output-file ${output_prefix}-${cls_input}.json

accelerate launch --deepspeed_config_file train/deepspeed.json inference/train_btcls.py \
    --model_name /workspace/HFModels/Qwen2.5-3B-Instruct \
    --data_file ${output_prefix}-${cls_input}.json \
    --output_dir  /workspace/HFModels/Qwen2.5-3B-Instruct-RM-${cls_input} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --max_length 1024 \
    --logging_steps 10 \
    --save_strategy no \
    --gradient_checkpointing True \
    --bf16 True  # 直接在这里启用混合精度！ 