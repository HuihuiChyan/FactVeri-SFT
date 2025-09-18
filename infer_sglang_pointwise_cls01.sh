<<<<<<< HEAD
export CUDA_VISIBLE_DEVICES=6
=======
export CUDA_VISIBLE_DEVICES=7
>>>>>>> c28abd39de82311c8121c468ee7d402705868a74

model_path=/workspace/HFModels/
model_name=Qwen3-4B
mode=local_retrieval # choose between local_retrieval and direct_gen
scheme=pointwise
<<<<<<< HEAD
dataset=triviaqa
search_api=searxng
dataset_path=/workspace/FactVeri-SFT/corpora/$dataset
dataset_name_without_ext=$dataset\_selection
python -u inference/infer_batch_sglang.py \
    --model_path $model_path/$model_name \
    --input_file $dataset_path/$dataset_name_without_ext.jsonl \
    --output_file ./results/$dataset_name_without_ext-$model_name-$scheme-$search_api.json \
    --mode $mode \
    --scheme $scheme \
    --disable_thinking \
    --search_api $search_api
=======
<<<<<<<< HEAD:infer_sglang_pointwise_cls02.sh
dataset=triviaqa
search_api=searxng
========
dataset=2wiki
>>>>>>>> c28abd39de82311c8121c468ee7d402705868a74:infer_sglang_pointwise_cls01.sh
dataset_path=/workspace/FactVeri-SFT/corpora/$dataset
dataset_name_without_ext=$dataset\_selection
# python -u inference/infer_batch_sglang.py \
#     --model_path $model_path/$model_name \
#     --input_file $dataset_path/$dataset_name_without_ext.jsonl \
#     --output_file ./results/$dataset_name_without_ext-$model_name-$scheme-$search_api.json \
#     --mode $mode \
#     --scheme $scheme \
#     --disable_thinking \
#     --search_api $search_api
>>>>>>> c28abd39de82311c8121c468ee7d402705868a74

cls_input_mode=trace
infer_model_path=/workspace/HFModels
infer_model_name=Qwen2.5-3B-Instruct-RM-$cls_input_mode/final_model

python -u inference/infer_classifier.py \
    --model_path $infer_model_path/$infer_model_name \
    --cls-input $cls_input_mode \
    --lora_path /workspace/HFModels/Qwen2.5-0.5B-Instruct-RM-trace-LoRA/lora_weights \
    --input_file ./results/$dataset_name_without_ext-$model_name-$scheme-$search_api.json \
    --output_file ./results/$dataset_name_without_ext-$model_name-$scheme-$search_api-cls.json