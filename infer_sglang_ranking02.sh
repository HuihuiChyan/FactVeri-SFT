export CUDA_VISIBLE_DEVICES=6

model_path=/workspace/HFModels/
model_name=Qwen3-8B
mode=direct_gen # choose between local_retrieval and direct_gen
scheme=ranking
dataset_path=/workspace/FactVeri-SFT/corpora/musique
dataset_name_without_ext=musique_verification
python -u inference/infer_batch_sglang.py \
    --model_path $model_path/$model_name \
    --input_file $dataset_path/$dataset_name_without_ext.jsonl \
    --output_file ./results/$dataset_name_without_ext-$model_name-$mode-$scheme.json \
    --mode $mode \
    --scheme $scheme \
    --disable_thinking