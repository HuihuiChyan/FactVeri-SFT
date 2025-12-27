export CUDA_VISIBLE_DEVICES=0

export SERPER_KEY_PRIVATE=95cc94f4818a2ffbc6b80a3c935d5729a24a087f

model_name=gpt-4.1
mode=retrieval # choose between retrieval and direct_gen
scheme=ranking
dataset=triviaqa_new
dataset_path=/workspace/FactVeri-SFT/corpora/$dataset
dataset_name_without_ext=$dataset\_verification
python -u src/infer_batch_gpt4.py \
    --model_name $model_name \
    --input_file $dataset_path/$dataset_name_without_ext.jsonl \
    --output_file $dataset_path/$dataset_name_without_ext-$model_name-$mode-$scheme.json \
    --mode $mode