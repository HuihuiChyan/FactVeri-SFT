import argparse
import json
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
)
from typing import List, Dict
from scipy.stats import kendalltau
from process_cls_input import create_prompt_for_cls
# 导入peft库以支持LoRA
from peft import PeftModel
from infer_batch_sglang import evaluate_final_results_ranking

def main():
    """主函数，运行分类和评估。"""
    parser = argparse.ArgumentParser(
        description="使用分类器对提取的事实进行评估，以验证事实。"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="序列分类模型或基础模型的路径。",
    )
    # --- 新增LoRA相关参数 ---
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="LoRA适配器的路径 (仅在 --use_lora 启用时使用)。",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true", # 当命令行包含 --use_lora 时，此参数为 True
        help="启用LoRA进行推理。",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--cls_input",
        type=str,
        choices=("direct_input", "full_history", "sum_history", "only_facts"),
        default="full_history"
    )
    args = parser.parse_args()

    print("加载分类器模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # --- 修改模型加载逻辑以支持LoRA ---
    if args.use_lora:
        if not args.lora_path:
            raise ValueError("使用 --use_lora 时必须指定 --lora_path。")
        print(f"从 {args.model_path} 加载基础模型...")
        base_model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=1)
        print(f"从 {args.lora_path} 加载LoRA适配器...")
        model = PeftModel.from_pretrained(base_model, args.lora_path)
        print("合并LoRA权重...")
        model = model.merge_and_unload() # 合并权重以加速推理
        model.cuda()
        print("LoRA模型加载并合并完成。")
    else:
        print(f"从 {args.model_path} 加载完整模型...")
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path).cuda()
        print("完整模型加载完成。")
    # ------------------------------------
    
    model.eval() # 设置模型为评估模式

    print(f"从 {args.input_file} 加载数据...")
    with open(args.input_file, "r", encoding="utf-8") as fin:
        lines = [json.loads(line) for line in fin]

    final_results = []
    with torch.no_grad():
        for item in tqdm.tqdm(lines, desc="分类答案"):
            answers = item["answers"]
            answer_scores = []

            for i, answer_item in enumerate(answers):
                # Create prompt using the specified cls_input mode
                conversation = create_prompt_for_cls(item, answer_item, args.cls_input, tokenizer)
                inputs = tokenizer(conversation, return_tensors="pt", truncation=True, max_length=args.max_length).to(model.device)
                
                # Get the raw logit score from the classifier
                score = model(**inputs).logits.item()
                answer_scores.append(score)
                # Optionally save the score for each answer
                answer_item["factuality_score"] = score

            # 预测结果是得分最高的答案的索引
            prediction = answer_scores.index(max(answer_scores))
            item["prediction"] = prediction
            predicted_ranking = sorted(
                range(len(answer_scores)), 
                key=lambda k: answer_scores[k], 
                reverse=True
            )
            predicted_ranking = [p+1 for p in predicted_ranking] # index starts from 1
            item["predicted_ranking"] = predicted_ranking
            final_results.append(item)

    print(f"将带有预测的结果保存到 {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for result in final_results:
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("开始最终评估...")
    ranking_metrics = evaluate_final_results_ranking(final_results)
    print(f"排序评估指标: {json.dumps(ranking_metrics, indent=4)}")

if __name__ == "__main__":
    main()
