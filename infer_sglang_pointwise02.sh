export CUDA_VISIBLE_DEVICES=2

model_path=/workspace/HFModels/
model_name=Qwen2.5-7B-Instruct
mode=local_retrieval # choose between local_retrieval and direct_gen
scheme=pointwise
dataset_path=/workspace/FactVeri-SFT/corpora/nq_hotpot_train_head
dataset_name_without_ext=nq_hotpot_train_1000_testset
python -u inference/infer_batch_sglang.py \
    --model_path $model_path/$model_name \
    --input_file $dataset_path/$dataset_name_without_ext.jsonl \
    --output_file ./results/$dataset_name_without_ext-$model_name-$mode-$scheme.json \
    --mode $mode \
    --scheme $scheme