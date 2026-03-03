# run_real_veriscore.py 使用说明

## 功能

使用完整的VeriScore流程处理hotpotqa数据，包括：
1. **Claim Extraction（提取claims）**：将每个answer拆分成atomic claims
2. **Evidence Retrieval（检索证据）**：为每个claim检索证据（使用Serper API，带缓存）
3. **Claim Verification（验证claims）**：验证每个claim是否正确
4. **计算Precision分数**：基于验证结果计算Precision分数

## 使用方法

```bash
python ablation/run_real_veriscore.py \
    --input_file /root/autodl-tmp/FactVeri-SFT/corpora/hotpotqa_new/hotpotqa_new_verification-verify.json \
    --output_file /root/autodl-tmp/FactVeri-SFT/corpora/hotpotqa_new/hotpotqa_new_verification-gpt-4o-real-veriscore.json \
    --model_name_extraction gpt-4o \
    --model_name_verification gpt-4o \
    --cache_dir ./cache
```

## 参数说明

- `--input_file`: 输入JSONL文件路径（必需）
- `--output_file`: 输出JSON文件路径（必需）
- `--model_name_extraction`: 用于claim extraction的模型名称（默认：gpt-4o）
- `--model_name_verification`: 用于claim verification的模型名称（默认：gpt-4o）
- `--cache_dir`: 缓存目录（默认：./cache）
- `--data_dir`: 数据目录，用于few-shot examples（默认：VeriScore/data）
- `--disable_cache`: 禁用缓存（默认：启用缓存）

## 输出格式

每个answer会包含以下字段：

### 详细claims信息：
- `claim_list`: 每个sentence对应的claims列表（二维列表）
- `all_claims`: 所有去重后的claims列表（一维列表）
- `claim_search_results`: 每个claim对应的搜索结果（字典格式）
- `claim_verification_result`: 每个claim的验证结果（列表）

### VeriScore综合分数：
- `veriscore_precision`: Precision分数（supported_claims / total_claims）
  - 当total_claims = 0时，设置为0.0
  - 当total_claims > 0时，计算supported_claims / total_claims
- `veriscore_stats`: 统计信息字典
  - `supported_claims`: 被支持的claims数量
  - `total_claims`: 总claims数量
  - `unsupported_claims`: 不被支持的claims数量
  - `sentence_count`: sentence数量

## 环境变量

需要设置以下环境变量：
- `OPENAI_API_KEY`: OpenAI API密钥
- `OPENAI_BASE_URL`: API基础URL
- `SERPER_KEY_PRIVATE`: Serper API密钥（用于Google搜索）

## 依赖项

- spacy模型：`en_core_web_sm`
  - 安装命令：`python -m spacy download en_core_web_sm`
- 其他Python包：tqdm, spacy等

## 注意事项

1. 输入文件格式：JSONL（每行一个JSON对象）
2. 输出文件格式：JSON（JSON数组）
3. 缓存文件保存在`cache/search_cache_serper.json`，可以加速重复查询
4. 处理大量数据时，注意API调用频率限制
