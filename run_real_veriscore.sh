export SERPER_KEY_PRIVATE=95cc94f4818a2ffbc6b80a3c935d5729a24a087f
export OPENAI_API_KEY=sk-XI5BH5GCT4jyfoYklI5BklGiJjX6C4Tw9M6UeAYVVu2xI2V5
export OPENAI_BASE_URL=https://yunwu.ai/v1

# python ablation/run_real_veriscore.py \
#     --input_file /root/autodl-tmp/FactVeri-SFT/corpora/hotpotqa_new/hotpotqa_new_verification-verify.json \
#     --output_file /root/autodl-tmp/FactVeri-SFT/corpora/hotpotqa_new/hotpotqa_new_verification-gpt-4o-real-veriscore.json \
#     --model_name_extraction gpt-4o \
#     --model_name_verification gpt-4o \
#     --cache_dir ./cache

python -u ablation/evaluate_real_veriscore.py \
    /root/autodl-tmp/FactVeri-SFT/corpora/hotpotqa_new/hotpotqa_new_verification-gpt-4o-real-veriscore.json