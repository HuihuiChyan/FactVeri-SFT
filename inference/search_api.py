from search_api_serper import SearchAPI as SerperSearchAPI
from search_api_searxng import SearchAPI as SearxngSearchAPI


class SearchAPI:
    """
    统一的搜索API接口，可以根据配置选择使用Serper或Searxng。
    """ 
    def __init__(self, search_url="https://google.serper.dev/search", search_type="serper"):
        """
        初始化搜索API
        
        Args:
            search_url: 搜索服务的URL
            api_type: 搜索API类型，可以是 "serper" 或 "searxng"
        """
        self.api_type = search_type.lower()
        
        if self.api_type == "serper":
            assert search_url == "https://google.serper.dev/search"
            self.search_api = SerperSearchAPI(search_url)
        elif self.api_type == "searxng":
            self.search_api = SearxngSearchAPI(search_url)
        else:
            raise ValueError(f"Unsupported API type: {search_type}. Supported types: 'serper', 'searxng'")
    
    def get_search_res(self, query):
        """
        执行搜索并返回格式化的结果字符串
        """
        return self.search_api.get_search_res(query)
    
    def save_cache(self):
        """
        保存缓存
        """
        self.search_api.save_cache()
    
    def load_cache(self):
        """
        加载缓存
        """
        return self.search_api.load_cache()
    
    @property
    def cache_file(self):
        """
        获取缓存文件路径
        """
        return self.search_api.cache_file
    
    @cache_file.setter
    def cache_file(self, value):
        """
        设置缓存文件路径
        """
        self.search_api.cache_file = value
    
    @property
    def cache_dict(self):
        """
        获取缓存字典
        """
        return self.search_api.cache_dict
    
    @cache_dict.setter
    def cache_dict(self, value):
        """
        设置缓存字典
        """
        self.search_api.cache_dict = value
