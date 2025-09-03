<<<<<<< HEAD
<<<<<<< HEAD
GPU_INDICES="4,5,6,7"
=======
GPU_INDICES="3,4,5,6"
>>>>>>> c28abd39de82311c8121c468ee7d402705868a74
=======
GPU_INDICES="3,4,5,6"
>>>>>>> c28abd3 (ready for combining evaluate and verdict)

# Model and dataset parameters
model_path="/workspace/HFModels/"
model_name="Qwen3-4B"
mode="local_retrieval" # Choose between "local_retrieval" and "direct_gen"
scheme="pointwise"     # Choose between "pointwise" and "best_of_n"
dataset_path="/workspace/FactVeri-SFT/corpora/nq_hotpot_train_head"
dataset_name_without_ext="nq_hotpot_train_head_selection"

# Construct full paths and prefixes
input_file="${dataset_path}/${dataset_name_without_ext}.jsonl"
output_prefix="./results/${dataset_name_without_ext}-${model_name}-${scheme}-${mode}"
merged_output_file="${output_prefix}.json"

# Create necessary directories
temp_dir="./temp_data"
results_dir="./results"
mkdir -p "$temp_dir"
mkdir -p "$results_dir"

echo "Input file: $input_file"
echo "Output prefix: $output_prefix"
echo "Temporary directory: $temp_dir"

# 1. Split GPU indices string into an array
IFS=',' read -r -a gpu_array <<< "$GPU_INDICES"
num_gpus=${#gpu_array[@]}

# 2. Split the input file into shards based on the number of GPUs
echo "Splitting input file into $num_gpus shards..."
total_lines=$(wc -l < "$input_file")
# Ceiling division to ensure all lines are covered
lines_per_shard=$(( (total_lines + num_gpus - 1) / num_gpus )) 
split -l "$lines_per_shard" -d --additional-suffix=.jsonl "$input_file" "$temp_dir/shard_"

# 3. Launch inference processes in the background for each GPU
echo "Starting inference on GPUs: $GPU_INDICES"
pids=() # Array to store process IDs
for i in "${!gpu_array[@]}"; do
    gpu_idx=${gpu_array[i]}
    shard_file=$(printf "%s/shard_%02d.jsonl" "$temp_dir" "$i")
    
    # Define output for this specific process
    output_file="${output_prefix}_gpu${gpu_idx}.jsonl"

    echo "Launching process on GPU ${gpu_idx} for shard ${shard_file}..."
    
    # Run the Python script in the background
    CUDA_VISIBLE_DEVICES=$gpu_idx python -u inference/infer_batch_sglang.py \
        --model_path "${model_path}/${model_name}" \
        --input_file "$shard_file" \
        --output_file "$output_file" \
        --mode "$mode" \
        --scheme "$scheme" \
        --disable_thinking &
    
    pids+=($!) # Store the PID of the background process
done

# 4. Wait for all background processes to complete
echo "Waiting for all inference processes to complete... PIDs: ${pids[*]}"
wait "${pids[@]}"
echo "All processes have completed."

# 5. Merge the result files from all GPUs
echo "Merging results..."
rm -f "$merged_output_file" # Remove old merged file if it exists

# Find, sort, and concatenate shard results to maintain order
find "$results_dir" -type f -wholename "${output_prefix}_gpu*.jsonl" | sort | xargs cat >> "$merged_output_file"

# 6. Clean up temporary and intermediate files
echo "Cleaning up temporary files..."
rm -rf "$temp_dir"
find "$results_dir" -type f -wholename "${output_prefix}_gpu*.jsonl" -delete

echo "---"
echo "Parallel inference complete!"
echo "Final merged output is located at: $merged_output_file"
echo "---"