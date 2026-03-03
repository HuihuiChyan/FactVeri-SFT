"""
适配器：将FactVeri-SFT的request_gpt适配为VeriScore的GetResponse接口
"""
import os
import sys
import json
import tiktoken

# 添加FactVeri-SFT/src到路径
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ABLATION_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_REPO_ROOT = os.path.dirname(_ABLATION_DIR)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from infer_batch_gpt4 import request_gpt


class GetResponseAdapter:
    """
    适配器类，将request_gpt适配为VeriScore期望的GetResponse接口
    """
    def __init__(self, cache_file, model_name="gpt-4o", max_tokens=1000, temperature=0):
        """
        初始化适配器
        Args:
            cache_file: 缓存文件路径
            model_name: 模型名称
            max_tokens: 最大token数
            temperature: 温度参数
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_file = cache_file
        
        # 使用tiktoken进行token计数
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        # 缓存相关
        self.cache_dict = self.load_cache()
        self.add_n = 0
        self.save_interval = 1
        self.print_interval = 20
    
    def get_response(self, system_message, prompt_text, cost_estimate_only=False):
        """
        获取模型响应
        Args:
            system_message: 系统消息
            prompt_text: 提示文本
            cost_estimate_only: 是否仅估算成本
        Returns:
            response_content: 响应内容
            prompt_tokens: prompt token数
            response_tokens: response token数
        """
        prompt_tokens = len(self.tokenizer.encode(prompt_text))
        if cost_estimate_only:
            response_tokens = 0
            return None, prompt_tokens, response_tokens
        
        # 检查缓存
        cache_key = prompt_text.strip()
        if cache_key in self.cache_dict:
            cached_response = self.cache_dict[cache_key]
            return cached_response, 0, 0
        
        # 构建消息
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt_text}
        ]
        
        # 调用request_gpt
        try:
            response_message = request_gpt(
                messages=messages,
                model=self.model_name,
                temperature=self.temperature,
                tools=None
            )
            response_content = response_message.content or ""
        except Exception as e:
            print(f"Error calling request_gpt: {e}")
            response_content = ""
        
        # 更新缓存
        self.cache_dict[cache_key] = response_content.strip()
        self.add_n += 1
        
        # 保存缓存
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        if self.add_n % self.print_interval == 0:
            print(f"Saving # {self.add_n} cache to {self.cache_file}...")
        
        response_tokens = len(self.tokenizer.encode(response_content))
        return response_content, prompt_tokens, response_tokens
    
    def tok_count(self, text: str) -> int:
        """计算文本的token数"""
        return len(self.tokenizer.encode(text))
    
    def save_cache(self):
        """保存缓存"""
        # 加载最新缓存，合并更新
        existing_cache = self.load_cache()
        existing_cache.update(self.cache_dict)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        
        with open(self.cache_file, "w") as f:
            json.dump(existing_cache, f, indent=4)
    
    def load_cache(self):
        """加载缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    cache = json.load(f)
                    print(f"Loading cache from {self.cache_file}...")
                    return cache
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Could not load cache from {self.cache_file}. Starting with empty cache.")
                return {}
        else:
            return {}
