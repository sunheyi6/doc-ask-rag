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
    
    # DashScope 批量 Embedding 的每批文本条数（text-embedding-v1 接口限制）
    BATCH_SIZE = 10
    
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or config.EMBEDDING_MODEL
        self.api_key = api_key or config.DASHSCOPE_API_KEY
        if self.api_key:
            dashscope.api_key = self.api_key
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """对文档列表进行向量化（批量调用，减少 API 往返次数）"""
        if not texts:
            return []
        embeddings: List[List[float]] = []
        # 分批调用：一次请求传多段文本，而非逐段请求
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i:i + self.BATCH_SIZE]
            response = dashscope.TextEmbedding.call(
                model=self.model,
                input=batch,  # 批量输入：SDK 支持 list
            )
            if response.status_code == 200:
                # output.embeddings 顺序与输入 batch 顺序一致
                for item in response.output['embeddings']:
                    embeddings.append(item['embedding'])
            else:
                raise RuntimeError(f"Embedding failed: {response.message}")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """对单个查询进行向量化"""
        return self.embed_documents([text])[0]
