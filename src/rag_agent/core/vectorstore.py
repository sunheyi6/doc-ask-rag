"""
向量数据库模块 - 封装 ChromaDB 操作
"""
import os
import shutil
from typing import List, Tuple, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from .embeddings import DashScopeEmbeddings
from ..config import config


class VectorStore:
    """向量数据库管理器"""
    
    def __init__(self, persist_directory: str = None, collection_name: str = "documents"):
        self.persist_directory = persist_directory or config.CHROMA_PERSIST_DIR
        self.collection_name = collection_name
        self._vectorstore: Optional[Chroma] = None
        self._embeddings = DashScopeEmbeddings()
    
    def create_from_documents(
        self, 
        documents: List[Document],
        clear_existing: bool = False
    ) -> Chroma:
        """
        从文档创建向量库
        
        Args:
            documents: 文档列表
            clear_existing: 是否清空现有向量库
            
        Returns:
            Chroma 向量库实例
        """
        if clear_existing and os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self._embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
            collection_name=self.collection_name
        )
        return self._vectorstore
    
    def load(self) -> Optional[Chroma]:
        """加载已存在的向量库"""
        if not os.path.exists(self.persist_directory):
            return None
        
        self._vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self._embeddings,
            collection_name=self.collection_name
        )
        return self._vectorstore
    
    def similarity_search(
        self, 
        query: str, 
        k: int = None,
        score_threshold: float = None
    ) -> List[Tuple[Document, float]]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 相似度阈值（余弦距离，越小越相似）
            
        Returns:
            (Document, score) 元组列表
        """
        if self._vectorstore is None:
            raise RuntimeError("向量库未初始化，请先调用 create_from_documents 或 load")
        
        k = k or config.RETRIEVAL_TOP_K
        
        # 使用相似度搜索 with score
        docs_with_scores = self._vectorstore.similarity_search_with_score(query, k=k)
        
        # 过滤低于阈值的
        if score_threshold is not None:
            docs_with_scores = [
                (doc, score) for doc, score in docs_with_scores 
                if score <= score_threshold
            ]
        
        return docs_with_scores
    
    def get_relevant_context(
        self, 
        query: str, 
        top_k: int = 3,
        score_threshold: float = None
    ) -> str:
        """
        获取相关上下文文本（用于 RAG）
        
        Args:
            query: 查询文本
            top_k: 取前 k 个相关文档
            score_threshold: 相似度阈值
            
        Returns:
            合并的上下文文本
        """
        docs_with_scores = self.similarity_search(query, k=top_k * 2)
        
        threshold = score_threshold or config.SIMILARITY_THRESHOLD
        
        # 过滤并取前 top_k 个
        filtered_docs = [
            doc for doc, score in docs_with_scores 
            if score <= threshold
        ][:top_k]
        
        if not filtered_docs:
            return ""
        
        # 合并上下文
        contexts = [doc.page_content for doc in filtered_docs]
        return "\n\n---\n\n".join(contexts)
    
    def clear(self):
        """清空向量库"""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        self._vectorstore = None
    
    @property
    def vectorstore(self) -> Optional[Chroma]:
        return self._vectorstore
