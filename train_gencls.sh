export CUDA_VISIBLE_DEVICES=0,1

INPUT_MODEL="/workspace/HFModels/Qwen2.5-3B-Instruct"
INPUT=nq_hotpot_train_head_pairwise-Qwen2.5-7B-Instruct-local_retrieval-pairwise_full_trace
PROMPT_TEMPLATE="qwen"
OUTPUT_DIR=$INPUT_MODEL-$INPUT
LENGTH=4096

accelerate launch --main_process_port 0 src/train.py \
    --stage sft \
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