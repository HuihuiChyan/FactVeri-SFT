export CUDA_VISIBLE_DEVICES=4,6
accelerate launch --num_processes 2 --mixed_precision bf16 --deepspeed_config_file train/deepspeed.json \
    inference/train_btcls.py \
    --data_file /workspace/FactVeri-SFT/results/nq_hotpot_train_head_selection-Qwen2.5-7B-Instruct-pointwise-local_retrieval-train.json \
    --model_name /workspace/HFModels/Llama-3.2-3B-Instruct \
    --train_batch_size 1 \
    --output_dir /workspace/HFModels/Llama-3.2-3B-Instruct-RM-reformat-trace \
    --gradient_accumulation_steps 32 \
    --max_length 128