"""
从 Hugging Face 下载 wiki-18 索引与语料。
大文件易遇 Connection reset，脚本会自动重试。若需用普通 HTTP（慢但更稳），可先设置：
  export HF_HUB_DISABLE_XET=1
"""
import argparse
import os
import time

from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(description="Download files from a Hugging Face dataset repository.")
parser.add_argument("--repo_id", type=str, default="PeterJinGo/wiki-18-e5-index", help="Hugging Face repository ID")
parser.add_argument("--save_path", type=str, required=True, help="Local directory to save files")
parser.add_argument("--max_retries", type=int, default=5, help="Max retries per file on connection error")
args = parser.parse_args()

save_path = args.save_path
max_retries = args.max_retries


def download_with_retry(repo_id: str, filename: str, repo_type: str = "dataset"):
    for attempt in range(max_retries):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                local_dir=save_path,
            )
            return
        except (RuntimeError, OSError, ConnectionError) as e:
            print(f"Error downloading {repo_id}/{filename}: {e}")
            err_msg = str(e).lower()
            if "connection reset" in err_msg or "104" in err_msg or "connection" in err_msg or "reset" in err_msg:
                wait = min(60 * (2 ** attempt), 300)
                print(f"\n[Retry {attempt + 1}/{max_retries}] {filename} failed: {type(e).__name__}. Retrying in {wait}s ...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed to download {repo_id}/{filename} after {max_retries} retries.")


# # wiki-18-e5-index (大文件 part_aa ~43GB, part_ab)
# repo_id = "PeterJinGo/wiki-18-e5-index"
# for file in ["part_aa", "part_ab"]:
#     download_with_retry(repo_id, file)

# wiki-18-corpus
print("Downloading wiki-18-corpus...")
repo_id = "PeterJinGo/wiki-18-corpus"
download_with_retry(repo_id, "wiki-18.jsonl.gz")