export CUDA_VISIBLE_DEVICES=7
export SERPER_KEY_PRIVATE=0325f2478ebae737e125dafc8d94de5334af1e8d

model_path=/workspace/HFModels/
model_name=Qwen3-4B
mode=local_retrieval # choose between local_retrieval and direct_gen
scheme=best_of_n
dataset_path=/workspace/FactVeri-SFT/corpora/musique
dataset_name_without_ext=musique_selection
python -u inference/infer_batch_sglang.py \
    --model_path $model_path/$model_name \
    --input_file $dataset_path/$dataset_name_without_ext.jsonl \
    --output_file ./results/$dataset_name_without_ext-$model_name-$mode-$scheme.json \
    --mode $mode \
    --scheme $scheme \
    --search_api searxng