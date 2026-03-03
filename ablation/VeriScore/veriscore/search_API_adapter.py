"""
适配器：将FactVeri-SFT的SearchAPISerper适配为VeriScore的SearchAPI接口
"""
import os
import sys

# 添加FactVeri-SFT/src到路径
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ABLATION_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_REPO_ROOT = os.path.dirname(_ABLATION_DIR)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from search_api_serper import SearchAPISerper


class SearchAPIAdapter:
    """
    适配器类，将SearchAPISerper适配为VeriScore期望的SearchAPI接口
    """
    def __init__(self, use_cache=True, **kwargs):
        """
        初始化适配器
        Args:
            use_cache: 是否使用缓存（默认True）
            **kwargs: 其他参数（为了兼容性保留）
        """
        # 默认使用缓存
        self.serper_api = SearchAPISerper(use_cache=use_cache)
        self.add_n = 0
        self.save_interval = 10
        self.cache_file = "cache/search_cache_serper.json"  # 为了兼容性
    
    def get_snippets(self, claim_lst):
        """
        为每个claim获取搜索结果snippets
        Args:
            claim_lst: claim列表
        Returns:
            text_claim_snippets_dict: 字典，key为claim，value为搜索结果列表
        """
        text_claim_snippets_dict = {}
        for query in claim_lst:
            # 使用SearchAPISerper的search_api_call方法进行批量搜索
            # 但我们需要单个查询的结果，所以直接调用get_search_res
            formatted_result = self.serper_api.get_search_res(query)
            
            # 解析格式化的结果字符串，提取搜索结果
            # formatted_result格式: "Result 1: [Title: ...] [Snippet: ...] [URL: ...]\nResult 2: ..."
            search_res_lst = self._parse_formatted_results(formatted_result)
            text_claim_snippets_dict[query] = search_res_lst
        
        return text_claim_snippets_dict
    
    def _parse_formatted_results(self, formatted_result):
        """
        解析格式化的搜索结果字符串，提取title, snippet, link
        Args:
            formatted_result: 格式化的搜索结果字符串
        Returns:
            search_res_lst: 搜索结果列表，每个元素包含title, snippet, link
        """
        search_res_lst = []
        if not formatted_result or formatted_result == "No results found.":
            return search_res_lst
        
        # 按行分割结果
        lines = formatted_result.strip().split('\n')
        for line in lines:
            if not line.strip() or not line.startswith('Result'):
                continue
            
            # 解析格式: "Result X: [Title: ...] [Snippet: ...] [URL: ...]"
            title = ""
            snippet = ""
            link = ""
            
            # 提取Title
            if '[Title:' in line:
                title_start = line.find('[Title:') + 7
                title_end = line.find(']', title_start)
                if title_end > title_start:
                    title = line[title_start:title_end].strip()
            
            # 提取Snippet
            if '[Snippet:' in line:
                snippet_start = line.find('[Snippet:') + 9
                snippet_end = line.find(']', snippet_start)
                if snippet_end > snippet_start:
                    snippet = line[snippet_start:snippet_end].strip()
            
            # 提取URL
            if '[URL:' in line:
                url_start = line.find('[URL:') + 5
                url_end = line.find(']', url_start)
                if url_end > url_start:
                    link = line[url_start:url_end].strip()
            
            if title or snippet or link:
                search_res_lst.append({
                    "title": title,
                    "snippet": snippet,
                    "link": link
                })
        
        return search_res_lst
    
    def get_search_res(self, query):
        """
        获取原始搜索结果（为了兼容性保留）
        Args:
            query: 搜索查询
        Returns:
            dict: 包含organic字段的字典
        """
        formatted_result = self.serper_api.get_search_res(query)
        search_res_lst = self._parse_formatted_results(formatted_result)
        
        # 转换为VeriScore期望的格式
        return {
            "organic": [
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", "")
                }
                for item in search_res_lst
            ]
        }
    
    def save_cache(self):
        """保存缓存（由SearchAPISerper内部处理）"""
        pass
    
    def load_cache(self):
        """加载缓存（由SearchAPISerper内部处理）"""
        return {}
