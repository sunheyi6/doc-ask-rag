"""
Embedding 模块 - 封装 DashScope Embedding 服务
"""
from typing import List
import dashscope
from langchain_core.embeddings import Embeddings
from ..config import config


class DashScopeEmbeddings(Embeddings):
    """
    使用 dashscope SDK 调用通义千问 Embedding 模型
    兼容 LangChain 的 Embeddings 接口
    """
    
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or config.EMBEDDING_MODEL
        self.api_key = api_key or config.DASHSCOPE_API_KEY
        if self.api_key:
            dashscope.api_key = self.api_key
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """对文档列表进行向量化"""
        responses = []
        for text in texts:
            response = dashscope.TextEmbedding.call(
                model=self.model,
                input=text
            )
            if response.status_code == 200:
                responses.append(response.output['embeddings'][0]['encoding'])
            else:
                raise RuntimeError(f"Embedding failed: {response.message}")
        return responses
    
    def embed_query(self, text: str) -> List[float]:
        """对单个查询进行向量化"""
        return self.embed_documents([text])[0]
