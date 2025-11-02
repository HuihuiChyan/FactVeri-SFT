export CUDA_VISIBLE_DEVICES=0

export SERPER_KEY_PRIVATE=95cc94f4818a2ffbc6b80a3c935d5729a24a087f

model_path=/workspace/HFModels/
model_name=Qwen2.5-3B-Instruct
mode=retrieval # choose between retrieval and direct_gen
scheme=ranking
dataset=nq_test_new
dataset_path=/workspace/FactVeri-SFT/corpora/$dataset
dataset_name_without_ext=$dataset\_verification
python -u inference/infer_batch_sglang.py \
    --model_path $model_path/$model_name \
    --input_file $dataset_path/$dataset_name_without_ext.jsonl \
    --output_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme.json \
    --mode $mode \
    --scheme $scheme