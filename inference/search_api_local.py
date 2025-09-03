import requests

class SearchAPILocal:
    """
    统一的搜索API接口，可以根据配置选择使用Serper或Searxng。
    """ 
    def __init__(self, search_url= "http://127.0.0.1:8000/retrieve", search_type="serper"):
        self.search_url = search_url
    
    def search_api_call(self, queries):
        if not queries:
            return []
        payload = {"queries": queries, "topk": 3, "return_scores": True}
        try:
            response = requests.post(
                self.search_url, json=payload, timeout=2000
            )
            response.raise_for_status()
            results_list = response.json()["result"]
        except requests.RequestException as e:
            print(f"Search API request failed for queries '{queries}': {e}")
            return ["Search failed due to connection error."] * len(queries)

        def _passages2string(retrieval_result):
            format_reference = ""
            for idx, doc_item in enumerate(retrieval_result):
                content = doc_item.get("document", {}).get("contents", "")
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            return format_reference

        return [_passages2string(results) for results in results_list]