#!/bin/bash

# 定义 GPU 索引列表，以逗号分隔
GPU_INDICES="2,5,6"

# 定义其他参数
input_file="/workspace/FactVeri-data/nq_hotpot_train/nq_hotpot_train_head_test.jsonl"
model_path="/workspace/HFModels/Qwen2.5-7B-Instruct"
mode="local_retrieval"
output_prefix="./results/nq_hotpot_train_head_test_$mode" # 统一的前缀

# 将 GPU 索引字符串分割成数组
IFS=',' read -r -a gpu_array <<< "$GPU_INDICES"
num_gpus=${#gpu_array[@]}

# 临时目录，用于存放分片文件
temp_dir=./temp_data
mkdir -p $temp_dir
echo "Using temporary directory: $temp_dir"

# # 1. 根据 GPU 数量分割输入文件
# echo "Splitting input file into $num_gpus shards..."
# # 使用 wc -l 计算总行数，然后除以 GPU 数量来确定每个分片的文件大小
# total_lines=$(wc -l < "$input_file")
# lines_per_shard=$(( (total_lines + num_gpus - 1) / num_gpus ))
# split -l $(( $(wc -l < "$input_file") / num_gpus + 1 )) -d --additional-suffix=.jsonl "$input_file" "$temp_dir/shard_"

# # 2. 启动多个子进程，每个进程处理一个分片
# echo "Starting inference on GPUs: $GPU_INDICES"
# for i in "${!gpu_array[@]}"; do
#     gpu_idx=${gpu_array[i]}
#     shard_file=$(printf "%s/shard_%02d.jsonl" "$temp_dir" "$i")
    
#     # 为每个分片构建独立的输出文件路径
#     output_file="${output_prefix}_gpu${gpu_idx}.jsonl"
#     log_file="./log/log_gpu${gpu_idx}.log"

#     # 在后台运行推理脚本，并将分片文件作为输入
#     CUDA_VISIBLE_DEVICES=$gpu_idx python -u infer_batch_sglang.py \
#         --model_path "$model_path" \
#         --input_file "$shard_file" \
#         --output_file "$output_file" \
#         --mode "$mode" > $log_file &
# done

# # 3. 等待所有子进程完成
# echo "Waiting for all processes to complete..."
# wait

# 4. 合并所有分片后的结果文件
echo "All processes completed. Merging results..."
merged_output_file="${output_prefix}_merged.jsonl"
rm -f "$merged_output_file"

# 使用 find 和 xargs 按序合并，确保最终结果顺序正确
find ./ -type f -wholename "${output_prefix}_gpu*.jsonl" | sort | xargs cat >> "$merged_output_file"

# 清理临时文件和分片输出文件
echo "Cleaning up temporary files..."
# rm -rf "$temp_dir"
# rm ./results/${output_prefix}_gpu*.jsonl

echo "Merging complete. Final output is at: $merged_output_file"