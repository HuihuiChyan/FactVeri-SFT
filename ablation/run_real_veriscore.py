"""
使用完整VeriScore流程处理hotpotqa数据
包括：Claim Extraction -> Evidence Retrieval -> Claim Verification -> 计算Precision分数
"""
import argparse
import json
import os
import sys
from typing import List, Dict
from tqdm import tqdm

# 添加路径以便导入VeriScore模块
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VERISCORE_DIR = os.path.join(_SCRIPT_DIR, "VeriScore")
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC = os.path.join(_REPO_ROOT, "src")

# 添加VeriScore和src到路径
if _VERISCORE_DIR not in sys.path:
    sys.path.insert(0, _VERISCORE_DIR)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# 导入适配器
from veriscore.search_API_adapter import SearchAPIAdapter
from veriscore.get_response_adapter import GetResponseAdapter

# 修改VeriScore模块的导入，使其使用适配器
# 在导入之前，替换GetResponse和SearchAPI
import veriscore.get_response as get_response_module
import veriscore.search_API as search_api_module

# 保存原始类
_original_get_response = get_response_module.GetResponse
_original_search_api = search_api_module.SearchAPI

# 替换为适配器
get_response_module.GetResponse = GetResponseAdapter
search_api_module.SearchAPI = SearchAPIAdapter

# 现在导入VeriScore模块（它们会使用适配器）
import spacy
from veriscore.claim_extractor import ClaimExtractor
from veriscore.claim_verifier import ClaimVerifier


class VeriScoreProcessor:
    """
    使用完整VeriScore流程处理数据的处理器
    """
    def __init__(self, model_name_extraction='gpt-4o', model_name_verification='gpt-4o',
                 cache_dir='./cache', data_dir=None, use_cache=True):
        """
        初始化处理器
        Args:
            model_name_extraction: 用于claim extraction的模型名称
            model_name_verification: 用于claim verification的模型名称
            cache_dir: 缓存目录
            data_dir: 数据目录（用于few-shot examples）
            use_cache: 是否使用缓存
        """
        self.cache_dir = cache_dir
        self.data_dir = data_dir or os.path.join(_VERISCORE_DIR, "data")
        self.use_cache = use_cache
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化spacy
        try:
            self.spacy_nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Please install it:")
            print("python -m spacy download en_core_web_sm")
            raise
        
        # 保存当前工作目录
        original_cwd = os.getcwd()
        
        # 切换到VeriScore目录，以便ClaimExtractor和ClaimVerifier能找到prompt文件
        veriscore_base_dir = os.path.join(_VERISCORE_DIR)
        os.chdir(veriscore_base_dir)
        
        try:
            # 初始化ClaimExtractor（会自动使用适配器）
            self.claim_extractor = ClaimExtractor(
                model_name=model_name_extraction,
                cache_dir=os.path.abspath(cache_dir),  # 使用绝对路径
                use_external_model=False
            )
            
            # 初始化ClaimVerifier（会自动使用适配器）
            demon_dir = os.path.join(self.data_dir, 'demos')
            self.claim_verifier = ClaimVerifier(
                model_name=model_name_verification,
                label_n=3,  # 三元分类：supported, unsupported, not enough information
                cache_dir=os.path.abspath(cache_dir),  # 使用绝对路径
                demon_dir=demon_dir,
                use_external_model=False
            )
        finally:
            # 恢复原始工作目录
            os.chdir(original_cwd)
        
        # 初始化SearchAPI适配器（用于证据检索）
        self.search_api = SearchAPIAdapter(use_cache=use_cache)
    
    def process_one_answer(self, question, answer_text):
        """
        处理一个answer，执行完整的VeriScore流程
        Args:
            question: 问题文本
            answer_text: 答案文本
        Returns:
            dict: 包含所有VeriScore结果的字典
        """
        result = {
            "claim_list": [],
            "all_claims": [],
            "claim_search_results": {},
            "claim_verification_result": [],
            "veriscore_precision": 0.0,
            "veriscore_stats": {
                "supported_claims": 0,
                "total_claims": 0,
                "unsupported_claims": 0,
                "sentence_count": 0
            }
        }
        
        # 保存当前工作目录
        original_cwd = os.getcwd()
        veriscore_base_dir = os.path.join(_VERISCORE_DIR)
        
        try:
            # 切换到VeriScore目录，以便能读取prompt文件
            os.chdir(veriscore_base_dir)
            
            # 阶段1: Claim Extraction
            if question and question.strip():
                snippet_lst, claim_list, all_claims, prompt_tok_cnt, response_tok_cnt = \
                    self.claim_extractor.qa_scanner_extractor(question, answer_text)
            else:
                snippet_lst, claim_list, all_claims, prompt_tok_cnt, response_tok_cnt = \
                    self.claim_extractor.non_qa_scanner_extractor(answer_text)
        finally:
            # 恢复工作目录
            os.chdir(original_cwd)
        
        result["claim_list"] = claim_list
        result["all_claims"] = all_claims
        result["veriscore_stats"]["sentence_count"] = len(claim_list)
        
        # 如果没有claims，直接返回
        if not all_claims or all_claims == ["No verifiable claim."]:
            return result
        
        # 阶段2: Evidence Retrieval
        # 使用SearchAPIAdapter进行搜索（不需要切换目录）
        claim_search_results = self.search_api.get_snippets(all_claims)
        result["claim_search_results"] = claim_search_results
        
        # 阶段3: Claim Verification
        # 需要在VeriScore目录下执行，以便读取prompt文件
        try:
            os.chdir(veriscore_base_dir)
            claim_verification_result, prompt_tok_cnt, response_tok_cnt = \
                self.claim_verifier.verifying_claim(claim_search_results, search_res_num=5)
            result["claim_verification_result"] = claim_verification_result
        finally:
            os.chdir(original_cwd)
        
        # 阶段4: 计算Precision分数
        total_claims = len(all_claims)
        supported_claims = 0
        unsupported_claims = 0
        
        for claim_veri_res in claim_verification_result:
            if isinstance(claim_veri_res, dict):
                verification_result = claim_veri_res.get('verification_result', '')
                # verification_result是经过处理的字符串（lowercase，去掉#和.）
                # 可能是 "supported", "unsupported", "not enough information" 等
                if verification_result:
                    verification_result_lower = str(verification_result).lower().strip()
                    # 检查是否包含"supported"关键词（且不包含"unsupported"）
                    if "supported" in verification_result_lower:
                        if "unsupported" not in verification_result_lower:
                            supported_claims += 1
                        else:
                            unsupported_claims += 1
                    elif "unsupported" in verification_result_lower or "contradict" in verification_result_lower:
                        unsupported_claims += 1
                    # 其他情况（如"not enough information"）不计入supported或unsupported
        
        result["veriscore_stats"]["supported_claims"] = supported_claims
        result["veriscore_stats"]["total_claims"] = total_claims
        result["veriscore_stats"]["unsupported_claims"] = unsupported_claims
        
        # 计算Precision
        if total_claims > 0:
            result["veriscore_precision"] = supported_claims / total_claims
        else:
            result["veriscore_precision"] = 0.0  # 当total_claims=0时，设置为0
        
        return result


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="使用完整VeriScore流程处理hotpotqa数据"
    )
    parser.add_argument("--input_file", type=str, required=True,
                       help="输入JSONL文件路径")
    parser.add_argument("--output_file", type=str, required=True,
                       help="输出JSON文件路径")
    parser.add_argument("--model_name_extraction", type=str, default="gpt-4o",
                       help="用于claim extraction的模型名称")
    parser.add_argument("--model_name_verification", type=str, default="gpt-4o",
                       help="用于claim verification的模型名称")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                       help="缓存目录")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="数据目录（用于few-shot examples）")
    parser.add_argument("--disable_cache", action="store_true", default=False,
                       help="禁用缓存")
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    print(f"Arguments: {args}")
    
    # 初始化处理器
    processor = VeriScoreProcessor(
        model_name_extraction=args.model_name_extraction,
        model_name_verification=args.model_name_verification,
        cache_dir=args.cache_dir,
        data_dir=args.data_dir,
        use_cache=not args.disable_cache
    )
    
    # 读取输入文件
    print(f"Loading data from {args.input_file}...")
    input_data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                input_data.append(json.loads(line))
    
    print(f"Loaded {len(input_data)} samples")
    
    # 处理每个样本
    final_results = []
    for item_idx, item in enumerate(tqdm(input_data, desc="Processing samples")):
        question = item.get("question", "")
        answers = item.get("answers", [])
        
        # 创建结果对象
        result_item = {**item}
        
        # 处理每个answer
        for answer_idx, answer in enumerate(answers):
            answer_text = answer.get("answer", "")
            
            # 执行VeriScore流程
            veriscore_result = processor.process_one_answer(question, answer_text)
            
            # 将结果添加到answer中
            answer.update(veriscore_result)
        
        final_results.append(result_item)
    
    # 保存结果
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        json.dump(final_results, f_out, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {args.output_file}")
    print(f"Processed {len(final_results)} samples")


if __name__ == "__main__":
    main()
