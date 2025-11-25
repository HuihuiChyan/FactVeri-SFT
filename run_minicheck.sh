export CUDA_VISIBLE_DEVICES=0

model_name=Qwen2.5-7B-Instruct
mode=retrieval # choose between retrieval and direct_gen
scheme=scoring
dataset=2wiki_new
dataset_path=/workspace/FactVeri-SFT/corpora/$dataset
dataset_name_without_ext=$dataset\_verification

python -u ablation/run_minicheck.py \
    $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme-sum.json \