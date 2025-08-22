export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --num_processes 8 --deepspeed_config_file deepspeed.json \
    train_btcls.py \
    --data_file /workspace/FactVeri-SFT/results/nq_hotpot_train_head_pointwise-Qwen2.5-7B-Instruct-local_retrieval-pointwise_doc-train.jsonl \
    --model_name /workspace/HFModels/Qwen2.5-0.5B-Instruct \
    --train_batch_size 2 \
    --output_dir /workspace/HFModels/Qwen2.5-0.5B-Instruct-RM-doc-trace \
    --gradient_accumulation_steps 4