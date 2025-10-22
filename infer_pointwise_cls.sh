export CUDA_VISIBLE_DEVICES=0

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export SERPER_KEY_PRIVATE=95cc94f4818a2ffbc6b80a3c935d5729a24a087f

model_path=/data/huanghui/HFModels/
model_name=Qwen2.5-7B-Instruct
mode=retrieval # choose between retrieval and direct_gen
scheme=pointwise
dataset=hotpotqa
dataset_path=/data/huanghui/FactVeri-SFT/corpora/$dataset
dataset_name_without_ext=$dataset\_verification
# python -u inference/infer_batch_sglang.py \
#     --model_path $model_path/$model_name \
#     --input_file $dataset_path/$dataset_name_without_ext.jsonl \
#     --output_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme.json \
#     --mode $mode \
#     --scheme $scheme 

cls_input=only_facts # choose between full_history, sum_history and direct_input

# python -u inference/infer_sum_pointwise.py \
#     --model_path $model_path/$model_name \
#     --input_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme.json \
#     --output_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme-sum.json \

infer_model_path=/data/huanghui/HFModels/
infer_model_name=Qwen2.5-7B-Instruct-RM-$mode-$cls_input/final_model
lora_model_name=Qwen2.5-7B-Instruct-RM-$mode-$cls_input/lora_weights

python -u inference/infer_classifier.py \
    --model_path $infer_model_path/$infer_model_name \
    --cls_input $cls_input \
    --input_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme-sum.json \
    --output_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme-$cls_input-cls.json 

    # --lora_path $infer_model_path/$lora_model_name \
    # --use_lora \