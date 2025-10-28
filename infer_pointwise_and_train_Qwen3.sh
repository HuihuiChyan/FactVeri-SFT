export CUDA_VISIBLE_DEVICES=0,1,2,3

export SERPER_KEY_PRIVATE=95cc94f4818a2ffbc6b80a3c935d5729a24a087f

model_path=/workspace/HFModels/
model_name=Qwen3-4B-Instruct-2507
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

cls_input=only_facts # choose between full_history, sum_history and direct_input

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
    --num_processes 4 \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file examples/deepspeed/ds_z2_config.json \
    inference/train_btcls.py \
    --model_name $model_path/$model_name \
    --data_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme-$cls_input.json \
    --output_dir $model_path/$model_name-RM-$mode-$cls_input \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --max_length 1024 \
    --logging_steps 10 \
    --save_strategy epoch \
    --gradient_checkpointing False \
    --use_lora \
    --bf16

model_path=/workspace/HFModels/
model_name=Qwen3-4B-Instruct-2507
mode=retrieval # choose between retrieval and direct_gen
scheme=pointwise
dataset_path=/workspace/FactVeri-SFT/corpora/nq_hotpotqa_train
dataset_name_without_ext=nq_hotpotqa_train_verification

cls_input=direct_input # choose between full_history, sum_history and direct_input

python -u inference/process_cls_input.py \
    --input-file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme-sum-convert.json \
    --cls-input $cls_input \
    --output-file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme-$cls_input.json \
    --tokenizer-path $model_path/$model_name

accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file examples/deepspeed/ds_z2_config.json \
    inference/train_btcls.py \
    --model_name $model_path/$model_name \
    --data_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme-$cls_input.json \
    --output_dir $model_path/$model_name-RM-$mode-$cls_input \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --max_length 1024 \
    --logging_steps 10 \
    --save_strategy epoch \
    --gradient_checkpointing False \
    --use_lora \
    --bf16

model_path=/workspace/HFModels/
model_name=Qwen3-4B-Instruct-2507
mode=retrieval # choose between retrieval and direct_gen
scheme=pointwise
dataset_path=/workspace/FactVeri-SFT/corpora/nq_hotpotqa_train
dataset_name_without_ext=nq_hotpotqa_train_verification

cls_input=sum_history # choose between full_history, sum_history and direct_input

python -u inference/process_cls_input.py \
    --input-file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme-sum-convert.json \
    --cls-input $cls_input \
    --output-file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme-$cls_input.json \
    --tokenizer-path $model_path/$model_name

accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file examples/deepspeed/ds_z2_config.json \
    inference/train_btcls.py \
    --model_name $model_path/$model_name \
    --data_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme-$cls_input.json \
    --output_dir $model_path/$model_name-RM-$mode-$cls_input \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --max_length 1024 \
    --logging_steps 10 \
    --save_strategy epoch \
    --gradient_checkpointing False \
    --use_lora \
    --bf16

model_path=/workspace/HFModels/
model_name=Qwen3-4B-Instruct-2507
mode=retrieval # choose between retrieval and direct_gen
scheme=pointwise
dataset_path=/workspace/FactVeri-SFT/corpora/nq_hotpotqa_train
dataset_name_without_ext=nq_hotpotqa_train_verification

cls_input=full_history # choose between full_history, sum_history and direct_input

python -u inference/process_cls_input.py \
    --input-file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme-sum-convert.json \
    --cls-input $cls_input \
    --output-file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme-$cls_input.json \
    --tokenizer-path $model_path/$model_name

accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file examples/deepspeed/ds_z2_config.json \
    inference/train_btcls.py \
    --model_name $model_path/$model_name \
    --data_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme-$cls_input.json \
    --output_dir $model_path/$model_name-RM-$mode-$cls_input \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 5e-5 \
    --max_length 2048 \
    --logging_steps 10 \
    --save_strategy epoch \
    --gradient_checkpointing False \
    --use_lora \
    --bf16