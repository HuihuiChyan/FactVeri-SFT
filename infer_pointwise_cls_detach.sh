export CUDA_VISIBLE_DEVICES=3

export SERPER_KEY_PRIVATE=95cc94f4818a2ffbc6b80a3c935d5729a24a087f

model_path=/workspace/HFModels/
model_name=Qwen3-4B-Instruct-2507
mode=retrieval # choose between retrieval and direct_gen
scheme=scoring
dataset=factscore_veri
dataset_path=/workspace/FactVeri-SFT/corpora/$dataset
dataset_name_without_ext=$dataset
python -u src/infer_batch_sglang_with_detach.py \
    --model_path $model_path/$model_name \
    --input_file $dataset_path/$dataset_name_without_ext.jsonl \
    --output_file $dataset_path/$dataset_name_without_ext-detach-$model_name-$mode-$scheme.json \
    --mode $mode \
    --scheme $scheme 

cls_input=sum_history # choose between full_history, sum_history and direct_input
learning_rate=2e-4

python -u src/infer_sum_pointwise_with_detach.py \
    --model_path $model_path/$model_name \
    --input_file $dataset_path/$dataset_name_without_ext-detach-$model_name-$mode-$scheme.json\
    --output_file $dataset_path/$dataset_name_without_ext-detach-$model_name-$mode-$scheme-sum.json \

infer_model_path=/workspace/HFModels/
infer_model_name=$model_name-RM-$mode-$cls_input/final_model
lora_model_name=$model_name-RM-$mode-$cls_input/lora_weights

python -u src/infer_classifier_with_detach.py \
    --model_path $infer_model_path/$infer_model_name \
    --cls_input $cls_input \
    --input_file $dataset_path/$dataset_name_without_ext-detach-$model_name-$mode-$scheme-sum.json \
    --output_file $dataset_path/$dataset_name_without_ext-detach-$model_name-$mode-$scheme-$cls_input-cls.json \
    --lora_path $infer_model_path/$lora_model_name \
    --use_lora