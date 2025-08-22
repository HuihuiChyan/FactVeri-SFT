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
)
from datasets import load_dataset
import os
from dataclasses import dataclass
from typing import Any, Dict, List
import argparse


# --- 自定义 Data Collator ---
@dataclass
class PairwiseDataCollatorWithPadding:
    """
    自定义 Data Collator，它将成对的输入正确地组合和填充。
    """
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

# --- 自定义 Loss 计算 ---
# 使用 transformers.Trainer 的一个主要区别是，它期望模型输出损失。
# 我们需要一个包装函数来计算我们的配对损失。
# Alternatively, you can define a custom Trainer subclass.
# For simplicity, let's create a custom loss computation class.
class PairwiseLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        重写 compute_loss 方法以使用 MarginRankingLoss。
        """
        # 从输入中分离标签
        labels = inputs.pop("labels")

        # 获取模型的输出（logits）
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # logits 的形状是 (2 * batch_size, 1)。
        # 我们需要正确地将它们分成 response1 和 response2 的分数。
        batch_size = logits.shape[0] // 2

        # squeeze(-1) 将形状从 (N, 1) 变为 (N)
        logits_squeezed = logits.squeeze(-1)
        scores1 = logits_squeezed[:batch_size]
        scores2 = logits_squeezed[batch_size:]

        # 根据标签创建 MarginRankingLoss 的目标(target)
        # target = 1  表示 score1 应该 > score2
        # target = -1 表示 score2 应该 > score1
        target = torch.ones_like(scores1)
        # 如果标签是 1 (即 response2 是正确的)，则 target 设为 -1
        target[labels == 1] = -1

        # 使用 MarginRankingLoss
        loss_fct = nn.MarginRankingLoss(margin=0.1)  # margin 是一个超参数
        loss = loss_fct(scores1, scores2, target)

        return (loss, outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser(description="使用 Accelerate Trainer 进行配对排序模型训练")

    # 定义命令行参数
    parser.add_argument("--model_name", type=str, default="Qwen2-5-3B-Instruct",
                        help="预训练模型的名称或路径")
    parser.add_argument("--data_file", type=str, default=None,
                        help="训练数据文件的路径")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="保存训练结果和模型的目录")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="训练的 epoch 数量")
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="每个设备的训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="每个设备的训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="学习率预热的步数")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--max_length", type=str, default=4096)
    
    args = parser.parse_args()

    # --- 数据加载与预处理 ---
    print("正在加载数据文件...")
    raw_datasets = load_dataset("json", data_files={"train": args.data_file})
    print("成功加载数据文件。")

    label2id = {"response1": 0, "response2": 1}

    print("正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    print("分词器加载完成。")

    def preprocess_function(examples):

        if 'pos_trace' in examples.keys():
            texts1 = examples["pos_trace"]
            texts2 = examples["neg_trace"]

            tokenized1 = tokenizer(texts1, max_length=args.max_length, truncation=True, return_attention_mask=False)
            tokenized2 = tokenizer(texts2, max_length=args.max_length, truncation=True, return_attention_mask=False)

            features = {}
            features["input_ids1"] = tokenized1["input_ids"]
            features["input_ids2"] = tokenized2["input_ids"]
            features["labels"] = examples["label"]
        
        else:
            questions = examples["question"]
            responses1 = examples["response1"]
            responses2 = examples["response2"]

            texts1 = [f"Question: {q} \n Response: {r}" for q, r in zip(questions, responses1)]
            texts2 = [f"Question: {q} \n Response: {r}" for q, r in zip(questions, responses2)]

            tokenized1 = tokenizer(texts1, max_length=args.max_length, truncation=True, return_attention_mask=False)
            tokenized2 = tokenizer(texts2, max_length=args.max_length, truncation=True, return_attention_mask=False)

            features = {}
            features["input_ids1"] = tokenized1["input_ids"]
            features["input_ids2"] = tokenized2["input_ids"]
            features["labels"] = [label2id[l] for l in examples["label"]]

        return features

    print("正在对数据集进行预处理...")
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    print("预处理完成。")

    # --- 模型加载 ---
    print(f"正在从 '{args.model_name}' 加载模型...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    print("模型加载完成。")

    # --- 训练参数配置 ---
    # `TrainingArguments` 包含了所有的训练配置，包括 Accelerate 和 FSDP 的配置
    # 注意，当与 accelerate launch 结合使用时，--per_device_train_batch_size 等参数可以由命令行直接传递
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=None,
        remove_unused_columns=False,  # 必须设置为 False，因为我们添加了额外的列
    )
    
    # --- 实例化 Trainer ---
    # Trainer 会自动处理数据加载器、优化器、调度器和 FSDP 的初始化
    trainer = PairwiseLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        data_collator=PairwiseDataCollatorWithPadding(tokenizer),
    )

    # --- 开始训练 ---
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    print("训练完成，模型已保存。")

if __name__ == "__main__":
    main()