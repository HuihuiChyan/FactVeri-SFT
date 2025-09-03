import json
import os
import requests
import time
import threading
import re
import tqdm
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor


class SearchAPISearxng:
    """
    一个用于与本地Searxng实例交互的搜索API类,
    此版本是线程安全的。
    """

    def __init__(self, searxng_url="http://host.docker.internal:8880"):
        # 本地searxng实例的基础URL。
        self.url = searxng_url

        # 为searxng结果设置缓存，以避免重复查询。
        self.cache_file = "cache/search_cache_serper.json"
        self.cache_dict = self.load_cache()

        # 线程安全机制
        self.cache_lock = threading.RLock()  # 使用可重入锁
        self.add_n = 0
        self.save_interval = 10
        print("Initialized SearchAPISearxng to use local searxng instance.")

    def format_search_results(self, search_json):
        """
        将来自 Searxng API 的JSON输出格式化为简洁的字符串。
        Searxng API 返回的结果在 'results' 键中，包含 'title', 'content', 'url' 字段。
        """
        if not search_json or not search_json.get("results"):
            return "No results found."

        formatted_str = ""
        results = search_json.get("results", [])
        for i, item in enumerate(results[:10]):  # 取前10个结果
            title = item.get("title", "No Title")
            snippet = item.get("content", "No Snippet")  # searxng使用'content'字段
            link = item.get("url", "#")  # searxng使用'url'字段

            # 移除换行符和多余空格
            clean_snippet = " ".join(snippet.replace("\n", " ").split())

            formatted_str += (
                f"Result {i+1}: [Title: {title}] [Snippet: {clean_snippet}] [URL: {link}]\n"
            )

        return formatted_str.strip()

    def get_search_res(self, query):
        """
        为给定的查询字符串查询本地searxng实例。
        以线程安全的方式实现缓存和无限重试循环。
        """
        cache_key = query.strip()

        # --- 首先，以线程安全的方式检查缓存 ---
        with self.cache_lock:
            if cache_key in self.cache_dict:
                cached_result = self.cache_dict[cache_key]
                # Return formatted results from cache
                return self.format_search_results(cached_result)

        # --- 如果不在缓存中，就在锁之外执行网络请求 ---
        # 这允许其他线程并发执行它们自己的请求。
        # '!google'前缀指定searxng内的搜索引擎。
        params = {"q": f"!google {query}", "format": "json", "language": "en"}

        response_json = None
        while True:
            try:
                response = requests.get(
                    self.url, params=params, verify=False, timeout=15
                )
                response.raise_for_status()  # 对错误的响应抛出HTTPError
                response_json = response.json()

                # 检查搜索引擎是否无响应
                unresponsive = response_json.get("unresponsive_engines", [])
                if not unresponsive:
                    break  # 成功，退出循环

                # 搜索引擎错误，重试
                # print(
                #     f"Query '{query}' failed due to unresponsive engines: {unresponsive}. Retrying in 3s..."
                # )
                time.sleep(3)

            except requests.exceptions.RequestException as e:
                # 这个'except'块处理searxng实例本身关闭的情况。
                # print(
                #     f"Could not connect to Searxng for query '{query}': {e}. Retrying in 5s..."
                # )
                time.sleep(5)  # 如果整个服务都关闭了，等待
                continue  # 重试连接

        # --- 查询成功后，在锁下更新并保存缓存 ---
        # 仅当最终结果有内容时才进行缓存
        if response_json:
            with self.cache_lock:
                self.cache_dict[cache_key] = response_json
                self.add_n += 1
                if self.add_n % self.save_interval == 0:
                    # 这个调用是安全的，因为我们使用的是RLock
                    self.save_cache()

        # Return formatted results as string
        final_result = response_json if response_json else {"results": []}
        return self.format_search_results(final_result)

    def save_cache(self):
        """
        以线程安全的方式将当前缓存字典保存到文件中。
        """
        with self.cache_lock:
            print(f"Saving searxng search cache...")
            # 写入前确保目录存在
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

            # 创建字典的副本以确保在转储过程中的一致性，
            # 即使锁提供了强有力的保护。
            cache_to_save = self.cache_dict.copy()

            with open(self.cache_file, "w") as f:
                json.dump(cache_to_save, f, indent=4)

    def load_cache(self):
        """如果缓存文件存在，则从中加载缓存。"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                try:
                    print(f"Loading searxng cache from {self.cache_file}...")
                    return json.load(f)
                except json.JSONDecodeError:
                    print(
                        f"Warning: Could not decode JSON from {self.cache_file}. Starting with an empty cache."
                    )
                    return {}
        else:
            return {}

    # def search_api_call(self, queries):
    #     if not queries:
    #         return []
        
    #     results = []
    #     for query in tqdm.tqdm(queries, desc="检索中"):
            
    #         result = self.get_search_res(query)
    #         results.append(result)

    #     return results
    
    def search_api_call(self, queries):
        if not queries:
            return []
        
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks to the executor and wrap with tqdm for a progress bar
            futures = [executor.submit(self.get_search_res, query) for query in queries]
            for future in tqdm.tqdm(futures, desc="检索中"):
                results.append(future.result())
        
        return results