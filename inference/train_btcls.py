# -*- coding: utf-8 -*-

# 1. 导入所需库
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
    TrainingArguments,
    Trainer,
    HfArgumentParser, # <--- 导入 HfArgumentParser
)
from datasets import load_dataset
import os
from dataclasses import dataclass, field # <--- 导入 field
from typing import Any, Dict, List, Optional # <--- 导入 Optional

# --- 自定义数据类来管理脚本参数 ---
@dataclass
class ScriptArguments:
    """
    定义脚本自身的参数，这些参数不属于 TrainingArguments。
    """
    model_name: str = field(metadata={"help": "预训练模型的名称或路径"})
    data_file: str = field(metadata={"help": "训练数据文件的路径"})
    max_length: int = field(default=4096, metadata={"help": "输入的最大长度"})

# --- 自定义 Data Collator (无需修改) ---
@dataclass
class PairwiseDataCollatorWithPadding:
    # ... (代码与原来完全相同)
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids1 = [f["input_ids1"] for f in features]
        input_ids2 = [f["input_ids2"] for f in features]
        labels = [f["labels"] for f in features]

        # 合并为一个 batch
        input_ids = input_ids1 + input_ids2
        batch = self.tokenizer.pad(
            [
                {"input_ids": ids} for ids in input_ids
            ],
            padding="longest",
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

# --- 自定义 Loss 计算 (无需修改) ---
class PairwiseLossTrainer(Trainer):
    # ... (代码与原来完全相同)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        batch_size = logits.shape[0] // 2
        logits_squeezed = logits.squeeze(-1)
        scores1 = logits_squeezed[:batch_size]
        scores2 = logits_squeezed[batch_size:]
        target = torch.ones_like(scores1)
        target[labels == 1] = -1
        loss_fct = nn.MarginRankingLoss(margin=0.1)
        loss = loss_fct(scores1, scores2, target)
        return (loss, outputs) if return_outputs else loss


def main():
    # --- 使用 HfArgumentParser 解析所有参数 ---
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # --- 数据加载与预处理 ---
    print("正在加载数据文件...")
    raw_datasets = load_dataset("json", data_files={"train": script_args.data_file})
    print("成功加载数据文件。")

    label2id = {"response1": 0, "response2": 1}

    print("正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True) # 添加 trust_remote_code 以支持新模型
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    print("分词器加载完成。")

    def preprocess_function(examples):
        # ... (内部逻辑与原来完全相同，只是把 args.max_length 改为 script_args.max_length)
        texts1 = examples["pos_trace"]
        texts2 = examples["neg_trace"]
        tokenized1 = tokenizer(texts1, max_length=script_args.max_length, truncation=True, return_attention_mask=False)
        tokenized2 = tokenizer(texts2, max_length=script_args.max_length, truncation=True, return_attention_mask=False)
        features = {}
        features["input_ids1"] = tokenized1["input_ids"]
        features["input_ids2"] = tokenized2["input_ids"]
        features["labels"] = examples["verify_result"]
        return features

    print("正在对数据集进行预处理...")
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    print("预处理完成。")

    # --- 模型加载 ---
    print(f"正在从 '{script_args.model_name}' 加载模型...")
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        num_labels=1,
        trust_remote_code=True # 添加 trust_remote_code
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    print("模型加载完成。")

    # --- 训练参数配置 (现在由 HfArgumentParser 自动完成) ---
    # 我们只需要添加一行 `remove_unused_columns=False`
    training_args.remove_unused_columns = False
    
    # --- 实例化 Trainer ---
    trainer = PairwiseLossTrainer(
        model=model,
        args=training_args, # 直接使用解析好的 training_args
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        data_collator=PairwiseDataCollatorWithPadding(tokenizer),
    )

    # --- 开始训练 ---
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))
    print("训练完成，模型已保存。")

if __name__ == "__main__":
    main()