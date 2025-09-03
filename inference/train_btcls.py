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
    HfArgumentParser,
)
from datasets import load_dataset
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from peft import LoraConfig, get_peft_model

# --- 自定义数据类来管理脚本参数 ---
@dataclass
class ScriptArguments:
    """
    定义脚本自身的参数，这些参数不属于 TrainingArguments。
    """
    model_name: str = field(metadata={"help": "预训练模型的名称或路径"})
    data_file: str = field(metadata={"help": "训练数据文件的路径"})
    max_length: int = field(default=4096, metadata={"help": "输入的最大长度"})
    use_lora: bool = field(default=False, metadata={"help": "是否开启LoRA训练"})
    lora_r: int = field(default=8, metadata={"help": "LoRA秩（控制参数规模）"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA缩放因子"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    target_modules: str = field(
        default="q_proj,k_proj,v_proj",
        metadata={"help": "LoRA目标层（Qwen的注意力层名）"},
    )
    num_labels: int = field(
        default=1, metadata={"help": "分类任务标签数（Pairwise任务固定为1）"}
    )

# -------------------------- 数据处理（Pairwise Collator） --------------------------
@dataclass
class PairwiseDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[str, bool] = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 分离正负样本特征
        input_ids1 = [f["input_ids1"] for f in features]
        attention_mask1 = [f["attention_mask1"] for f in features]
        input_ids2 = [f["input_ids2"] for f in features]
        attention_mask2 = [f["attention_mask2"] for f in features]
        labels = [f["labels"] for f in features]

        # 合并正负样本（正样本在前，负样本在后，总长度为 2B）
        all_input_ids = input_ids1 + input_ids2
        all_attention_mask = attention_mask1 + attention_mask2

        # 统一 Padding（确保所有 2B 条序列长度一致）
        batch = self.tokenizer.pad(
            {"input_ids": all_input_ids, "attention_mask": all_attention_mask},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # 添加标签（保持与样本对的对应关系）
        batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch

# --- 自定义 Loss 计算（Pairwise MarginRankingLoss） ---
class PairwiseLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"].squeeze(-1)  # 压缩到(batch_size,)

        # 分离正负样本的分数（前半是正样本，后半是负样本）
        batch_size = logits.shape[0] // 2
        scores_pos = logits[:batch_size]  # 正确样本分数
        scores_neg = logits[batch_size:]  # 错误样本分数

        # 构造损失目标：正确样本分数 > 错误样本分数（target=1）；反之target=-1
        target = torch.ones_like(scores_pos, dtype=torch.float32)
        target[labels == 1] = -1.0  # labels=1表示错误样本是正确的（需根据数据调整）

        # MarginRankingLoss：让正确样本分数比错误样本高至少0.1
        loss_fct = nn.MarginRankingLoss(margin=0.1)
        loss = loss_fct(scores_pos, scores_neg, target)

        return (loss, outputs) if return_outputs else loss


def main():
    # --- 使用 HfArgumentParser 解析所有参数 ---
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # 强制禁用自动移除未使用字段（关键！）
    training_args.remove_unused_columns = False

    # --- 数据加载与预处理 ---
    print("正在加载数据文件...")
    raw_datasets = load_dataset("json", data_files={"train": script_args.data_file})
    train_dataset = raw_datasets["train"]
    print("成功加载数据文件。")

    print("正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    print("分词器加载完成。")

    def preprocess_function(examples):

        pos_texts = examples["pos_trace"]  # 正确样本
        neg_texts = examples["neg_trace"]  # 错误样本

        # Tokenize正负样本（保留attention_mask）
        tokenized_pos = tokenizer(
            pos_texts,
            max_length=script_args.max_length,
            truncation=True,
            return_attention_mask=True,
        )
        tokenized_neg = tokenizer(
            neg_texts,
            max_length=script_args.max_length,
            truncation=True,
            return_attention_mask=True,
        )

        return {
            "input_ids1": tokenized_pos["input_ids"],
            "attention_mask1": tokenized_pos["attention_mask"],
            "input_ids2": tokenized_neg["input_ids"],
            "attention_mask2": tokenized_neg["attention_mask"],
            "labels": examples["verify_result"],  # 0=正确样本是pos，1=正确样本是neg
        }

    print("预处理数据...")
    tokenized_datasets = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="预处理训练数据",
    )

    # --- 模型加载 ---
    print(f"正在从 '{script_args.model_name}' 加载模型...")
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        num_labels=1,
        trust_remote_code=True # 添加 trust_remote_code
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    print("模型加载完成。")

    if script_args.use_lora:
        # 配置LoRA（仅训练Transformer的qkv层）
        lora_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            target_modules=script_args.target_modules.split(","),  # Qwen的注意力层名
            lora_dropout=script_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",  # 序列分类任务
            modules_to_save=["score"],
        )
        model = get_peft_model(model, lora_config)  # 应用LoRA
        print("可训练参数统计:")
        model.print_trainable_parameters()  # 输出：trainable params / all params / trainable%

    # model.base_model.model.score.original_module.requires_grad  # should be False
    # model.base_model.model.score.modules_to_save["default"].weight.requires_grad  # should be True
    # model.base_model.model.score.active_adapter  # should be 'default'
    # import pdb;pdb.set_trace()
    
    # --- 实例化 Trainer ---
    trainer = PairwiseLossTrainer(
        model=model,
        args=training_args, # 直接使用解析好的 training_args
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        data_collator=PairwiseDataCollatorWithPadding(
            tokenizer=tokenizer,
            max_length=script_args.max_length,
        ),
    )

    # --- 开始训练 ---
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))
    tokenizer.save_pretrained(training_args.output_dir)  # 保存Tokenizer
    if script_args.use_lora:
        lora_save_dir = os.path.join(training_args.output_dir, "lora_weights")
        model.save_pretrained(lora_save_dir) 
    
    print("训练完成，模型已保存。")

if __name__ == "__main__":
    main()