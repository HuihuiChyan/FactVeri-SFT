export CUDA_VISIBLE_DEVICES=1
model_path=/root/autodl-tmp/HFModels/
model_name=Qwen2.5-7B-Instruct

# SGLang 服务配置
sglang_url=http://localhost:30001
sglang_port=30001
sglang_host=0.0.0.0
tool_call_parser=qwen25  # 适用于 Qwen2.5 模型

# 启动 SGLang 服务（后台运行）
echo "Starting SGLang server..."
bash "./launch_sglang_server.sh" \
    --model_path "$model_path/$model_name" \
    --host "$sglang_host" \
    --port "$sglang_port" \
    --tool_call_parser "$tool_call_parser"