import json
import os
import re
import time
import threading
import requests
import tqdm
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor


class SearchAPISerper:
    def __init__(self, search_url="https://google.serper.dev/search"):
        # invariant variables
        self.serper_key = os.getenv("SERPER_KEY_PRIVATE")
        self.url = search_url
        self.headers = {'X-API-KEY': self.serper_key,
                        'Content-Type': 'application/json'}
        # cache related
        self.cache_file = "cache/search_cache_serper.json"
        self.cache_dict = self.load_cache()
        
        # 线程安全机制
        self.cache_lock = threading.RLock()  # 使用可重入锁
        self.add_n = 0
        self.save_interval = 10
        print("Initialized SearchAPISerper with thread-safe caching.")

    def format_search_results(self, search_json):
        """
        将来自 Serper API 的JSON输出格式化为简洁的字符串。
        Serper API 返回的结果在 'organic' 键中，包含 'title', 'snippet', 'link' 字段。
        """
        formatted_str = ""
        results = search_json.get("organic", [])
        for i, item in enumerate(results[:10]):  # 取前10个结果
            title = item.get("title", "No Title")
            snippet = item.get("snippet", "No Snippet")
            link = item.get("link", "#")

            # 移除换行符和多余空格
            clean_snippet = " ".join(snippet.replace("\n", " ").split())

            formatted_str += (
                f"Result {i+1}: [Title: {title}] [Snippet: {clean_snippet}] [URL: {link}]\n"
            )

        return formatted_str.strip()

    def get_search_res(self, query):
        """
        为给定的查询字符串查询Serper API。
        以线程安全的方式实现缓存。
        """
        query = query.strip().strip("\'").strip('\"')
        cache_key = query.strip()

        # --- 首先，以线程安全的方式检查缓存 ---
        with self.cache_lock:
            if cache_key in self.cache_dict:
                print("Loading cache from cache file!")
                cached_result = self.cache_dict[cache_key]
                # Return formatted results from cache
                return self.format_search_results(cached_result)

        # --- 如果不在缓存中，就在锁之外执行网络请求 ---
        # 这允许其他线程并发执行它们自己的请求。
        payload = json.dumps({"q": query})

        retry = 5
        for i in range(retry):
            response = requests.request("POST", self.url, headers=self.headers, data=payload)
            raw_result = literal_eval(response.text)
            if raw_result and raw_result.get("organic"):
                break
            elif "message" in raw_result and raw_result["message"] == "Not enough credits":
                raise Exception("Not enough credits, please re-charge!")
            else:
                # 并行模式下serper可能返回空内容，这时候需要多次retry
                time.sleep(5)

        # --- 查询成功后，在锁下更新并保存缓存 ---
        # 仅当最终结果有内容时才进行缓存
        if not raw_result.get("organic"):
            print(f"No results found for query: {query}.")
            return "No results found."
        elif not raw_result:
            print(f"Search result is None!")
            return "No results found."
        else:
            with self.cache_lock:
                self.cache_dict[cache_key] = raw_result
                self.add_n += 1
                if self.add_n % self.save_interval == 0:
                    # 这个调用是安全的，因为我们使用的是RLock
                    self.save_cache()
            # Return formatted results as string
            return self.format_search_results(raw_result)

    def save_cache(self):
        """
        以线程安全的方式将当前缓存字典保存到文件中。
        """
        with self.cache_lock:
            print(f"Saving serper search cache...")
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
                    print(f"Loading serper cache from {self.cache_file}...")
                    return json.load(f)
                except json.JSONDecodeError:
                    print(
                        f"Warning: Could not decode JSON from {self.cache_file}. Starting with an empty cache."
                    )
                    return {}
        else:
            return {}

    def search_api_call(self, queries):
        if not queries:
            return []
        
        # results = []
        # for query in tqdm.tqdm(queries, desc="Searching"):
        #     result = self.get_search_res(query)
        #     results.append(result)

        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks to the executor and wrap with tqdm for a progress bar
            futures = [executor.submit(self.get_search_res, query) for query in queries]
            for future in tqdm.tqdm(futures, desc="Searching"):
                results.append(future.result())

        return results