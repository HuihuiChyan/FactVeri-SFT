# Serving 目录说明

本目录包含单线程版本的 scoring 推理代码，通过 HTTP API 调用 SGLang 服务。

## 文件结构

- `infer_scoring.py`: 主推理脚本，单线程处理，通过 HTTP API 调用 SGLang 服务
- `search_api_local.py`: 本地文档检索 API（Wikipedia）
- `search_api_serper.py`: Serper 网络搜索 API（Google）
- `launch_sglang_server.sh`: 启动 SGLang HTTP 服务器的脚本
- `run_scoring.sh`: 测试脚本，参照 `infer_generative_scoring.sh` 的格式

## 使用流程

### 1. 启动 SGLang 服务

首先需要启动 SGLang HTTP 服务器：

```bash
./launch_sglang_server.sh --model_path /path/to/model [--port 30000] [--host 0.0.0.0]
```

或者直接使用 Python：

```bash
python3 -m sglang.launch_server --model-path /path/to/model --host 0.0.0.0 --port 30000
```

### 2. 运行推理

使用测试脚本运行推理：

```bash
./run_scoring.sh
```

或者直接使用 Python：

```bash
python serving/infer_scoring.py \
    --sglang_url http://localhost:30000 \
    --tokenizer_path /path/to/tokenizer \
    --input_file /path/to/input.jsonl \
    --output_file /path/to/output.json \
    --mode retrieval  # 或 direct_score
```

## 参数说明

### infer_scoring.py 参数

- `--sglang_url`: SGLang HTTP 服务器地址（默认: http://localhost:30000）
- `--tokenizer_path`: Tokenizer 路径（用于格式化 prompt）
- `--input_file`: 输入 JSONL 文件路径
- `--output_file`: 输出 JSON 文件路径
- `--mode`: 运行模式
  - `retrieval`: 先检索再评分
  - `direct_score`: 直接评分（无需检索）
- `--max_token`: 最大生成 token 数（默认: 2048）
- `--temperature`: 采样温度（默认: 0.0）
- `--disable_thinking`: 禁用模型的思考过程
- `--no-use-serper-cache`: 禁用 Serper 缓存

## 输入格式

输入文件应为 JSONL 格式，每行一个 JSON 对象：

```json
{
  "question": "问题文本",
  "answers": [
    {"model": "model1", "answer": "答案1"},
    {"model": "model2", "answer": "答案2"}
  ]
}
```

## 输出格式

输出文件为 JSON 格式，包含以下字段：

- `question`: 原始问题
- `answers`: 答案列表，每个答案包含：
  - `verdict_response`: 模型生成的评分响应
  - `predicted_scoring`: 提取的分数（1-10）
- `retrieval_path`: 检索历史（仅 retrieval 模式）
- `retrieval_stats`: 检索统计信息（仅 retrieval 模式）
- `e2e_latency_seconds`: 端到端延迟
- `total_sequence_tokens`: 总 token 数

## 注意事项

1. 确保 SGLang 服务已启动并运行在指定端口
2. 对于 retrieval 模式，需要设置 `SERPER_KEY_PRIVATE` 环境变量
3. 本地检索服务默认运行在 `http://127.0.0.1:8000/retrieve`
4. 缓存文件保存在 `serving/cache/` 目录下
