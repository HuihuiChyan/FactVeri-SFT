export CUDA_VISIBLE_DEVICES=0,1,6,7
accelerate launch --num_processes 4 --deepspeed_config_file deepspeed.json \
    train_btcls.py \
    --data_file /workspace/FactVeri-data/nq_hotpot_train_head/nq_hotpot_train_head_pairwise.jsonl \
    --model_name /workspace/HFModels/Qwen2.5-0.5B-Instruct \
    --train_batch_size 16 \
    --output_dir /workspace/HFModels/Qwen2.5-0.5B-Instruct-RM-no-trace