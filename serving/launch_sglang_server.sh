#!/bin/bash

# 启动 SGLang HTTP 服务器脚本
# 用法: ./launch_sglang_server.sh --model_path /path/to/model [--port 30000] [--host 0.0.0.0] [--tool_call_parser qwen25]

MODEL_PATH=""
PORT=30000
HOST="0.0.0.0"
TOOL_CALL_PARSER="qwen25"  # 默认使用 qwen25，适用于 Qwen2.5 模型

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --tool_call_parser)
            TOOL_CALL_PARSER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --model_path PATH [--port PORT] [--host HOST] [--tool_call_parser PARSER]"
            exit 1
            ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model_path is required"
    echo "Usage: $0 --model_path PATH [--port PORT] [--host HOST] [--tool_call_parser PARSER]"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

echo "Starting SGLang HTTP server..."
echo "Model path: $MODEL_PATH"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Tool call parser: $TOOL_CALL_PARSER"

python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tool-call-parser "$TOOL_CALL_PARSER"