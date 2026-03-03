# conda create -n retriever python=3.10
# conda activate retriever

# # we recommend installing torch with conda for faiss-gpu
# conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# pip install transformers datasets pyserini

# ## install the gpu version faiss to guarantee efficient RL rollout
# conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# ## API function
# pip install uvicorn fastapi

export HF_ENDPOINT=https://hf-mirror.com
# # 若 Xet 经常断连，可改用普通 HTTP（会变慢）：
# export HF_HUB_DISABLE_XET=1

save_path=./document
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz