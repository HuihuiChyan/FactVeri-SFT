export CUDA_VISIBLE_DEVICES=4,6

INPUT_MODEL="/workspace/HFModels/Qwen2.5-3B-Instruct"
INPUT=dpo_en_demo
PROMPT_TEMPLATE="qwen"
OUTPUT_DIR=$INPUT_MODEL-$INPUT
LENGTH=4096

accelerate launch --num_processes 2 src/train.py \
    --stage rm \
    --model_name_or_path=$INPUT_MODEL \
    --do_train \
    --dataset=${INPUT} \
    --template=$PROMPT_TEMPLATE \
    --finetuning_type full \
    --output_dir=$OUTPUT_DIR \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --num_train_epochs 2 \
    --learning_rate=5e-6 \
    --cutoff_len=$LENGTH \
    --preprocessing_num_workers=1 \
    --dataloader_num_workers=1 \
    --plot_loss \
    --deepspeed examples/deepspeed/ds_z2_offload_config.json \
    --report_to=none \
    --bf16