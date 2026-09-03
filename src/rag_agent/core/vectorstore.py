"""
向量数据库模块 - 封装 ChromaDB 操作
"""
import os
import shutil
import math
import re
from collections import Counter
from typing import List, Tuple, Optional, Callable
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from .embeddings import DashScopeEmbeddings
from ..config import config


def _tokenize(text: str) -> List[str]:
    """
    轻量中英文分词（无需第三方词库）：
    - 英文/数字：按单词切分
    - 中文：单字 + 2-gram（如“知识库” → 知识、识库），能捕捉常见关键词组合
    
    生产环境可替换为 jieba 等分词器，效果更好，但原理完全相同。
    """
    text = (text or "").lower()
    tokens: List[str] = []
    # 英文单词 / 数字
    tokens.extend(re.findall(r"[a-z0-9_]+", text))
    # 中文部分：单字 + 2-gram
    chars = "".join(re.findall(r"[\u4e00-\u9fff]+", text))
    tokens.extend(chars)
    if len(chars) >= 2:
        tokens.extend(chars[i:i + 2] for i in range(len(chars) - 1))
    return tokens


class BM25Index:
    """
    手写 BM25 关键词检索索引（零第三方依赖，便于学习核心公式）

    BM25 打分（对查询中每个词求和）：
        score(d, q) = Σ IDF(t) * tf(t,d)*(k1+1) / (tf(t,d) + k1*(1-b+b*|d|/avgdl))

    两个关键设计：
    1. IDF（逆文档频率）：词在越少的文档中出现，区分度越高，权重越大
    2. 词频饱和 + 长度归一化：词出现越多越重要但收益递减（k1 控制）；
       长文档天然词频高，需按长度惩罚（b 控制），避免“长文作弊”
    """

    def __init__(self, documents: List[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1  # 词频饱和度（越大，词频带来的收益衰减越慢）
        self.b = b    # 长度惩罚强度（0=不惩罚，1=全惩罚）
        self.tokenized_docs: List[List[str]] = [_tokenize(d.page_content) for d in documents]
        self.doc_lengths: List[int] = [len(t) for t in self.tokenized_docs]
        self.avg_doc_length: float = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        # 词 → 出现在多少篇文档中（df，用于 IDF）
        self.doc_freqs: dict = {}
        for tokens in self.tokenized_docs:
            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
        self.num_docs: int = len(documents)

    def _idf(self, term: str) -> float:
        """逆文档频率：词越稀有，权重越高"""
        df = self.doc_freqs.get(term, 0)
        return math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))

    def _score_doc(self, doc_index: int, query_terms: List[str]) -> float:
        """对单篇文档打分：BM25 公式"""
        doc_length = self.doc_lengths[doc_index]
        term_counts = Counter(self.tokenized_docs[doc_index])
        total = 0.0
        for term in query_terms:
            tf = term_counts.get(term, 0)
            if tf == 0:
                continue
            # 长度归一化分母：长文档的词频被惩罚
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            total += self._idf(term) * (tf * (self.k1 + 1)) / denominator
        return total

    def search(
        self,
        query: str,
        k: int = 10,
        metadata_predicate: Optional[Callable[[Document], bool]] = None,
    ) -> List[Tuple[Document, float]]:
        """关键词检索，按 BM25 分数降序返回前 k 条"""
        query_terms = _tokenize(query)
        if not query_terms:
            return []
        scored: List[Tuple[Document, float]] = []
        for i, doc in enumerate(self.documents):
            if metadata_predicate is not None and not metadata_predicate(doc):
                continue
            score = self._score_doc(i, query_terms)
            if score > 0:
                scored.append((doc, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]


def _rrf_fuse(
    ranked_lists: List[List[Tuple[Document, float]]],
    constant_k: int = 60,
) -> List[Tuple[Document, float]]:
    """
    Reciprocal Rank Fusion（RRF）：只利用“排名”融合多路检索结果
        score(d) = Σ 1 / (constant_k + rank_i(d))

    为什么用排名而不是分数？
    - 向量检索输出“余弦距离”（越小越好），BM25 输出“得分”（越大越好），
      两者尺度不同、不可直接比较，但“排名”是可以比的
    - constant_k=60 是论文建议的经验值，用于平滑排名带来的影响
    """
    fusion: dict = {}
    for ranked in ranked_lists:
        for rank, (doc, _) in enumerate(ranked):
            # 以“文件名+正文前 80 字”作为文档去重标识
            key = f"{doc.metadata.get('filename', '')}:{doc.page_content[:80]}"
            if key not in fusion:
                fusion[key] = [doc, 0.0]
            fusion[key][1] += 1.0 / (constant_k + rank + 1)
    result = sorted(fusion.values(), key=lambda item: item[1], reverse=True)
    return [(doc, score) for doc, score in result]


class VectorStore:
    """向量数据库管理器"""
    
    def __init__(self, persist_directory: str = None, collection_name: str = "documents"):
        self.persist_directory = persist_directory or config.CHROMA_PERSIST_DIR
        self.collection_name = collection_name
        self._vectorstore: Optional[Chroma] = None
        self._embeddings = DashScopeEmbeddings()
        self._bm25: Optional[BM25Index] = None
    
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
        # 同步构建 BM25 关键词索引（与向量索引共用同一批分块）
        self._bm25 = BM25Index(documents)
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
        # 从向量库取回全部分块，重建 BM25 索引（保持两种检索器数据一致）
        if self._bm25 is None:
            raw = self._vectorstore.get()
            documents = [
                Document(page_content=content, metadata=meta or {})
                for content, meta in zip(raw.get("documents", []), raw.get("metadatas", []))
            ]
            if documents:
                self._bm25 = BM25Index(documents)
        return self._vectorstore
    
    def similarity_search(
        self, 
        query: str, 
        k: int = None,
        score_threshold: float = None,
        metadata_filter: Optional[dict] = None
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
        docs_with_scores = self._vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=metadata_filter
        )
        
        # 过滤低于阈值的
        if score_threshold is not None:
            docs_with_scores = [
                (doc, score) for doc, score in docs_with_scores 
                if score <= score_threshold
            ]
        
        return docs_with_scores
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = None,
        score_threshold: float = None,
        metadata_filter: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        混合检索：向量语义检索 + BM25 关键词检索，RRF 融合排序
        
        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 向量侧相似度阈值（余弦距离，越小越相似）
                - 向量侧：超过阈值的视为“语义无关”，丢弃
                - BM25 侧：能命中关键词即视为相关，不受该阈值限制
            metadata_filter: 元数据过滤（如限定检索某个文件）
            
        Returns:
            (Document, 融合分数) 元组列表，分数越大越相关
        """
        k = k or config.RETRIEVAL_TOP_K
        threshold = score_threshold if score_threshold is not None else config.SIMILARITY_THRESHOLD
        candidate_k = max(k * 3, k + 1)  # 宽召回：先多取一些，融合后再精取 k 条
        
        # 1) 向量检索候选（语义门控）
        try:
            vector_candidates = [
                (doc, score) for doc, score in self.similarity_search(
                    query, k=candidate_k, metadata_filter=metadata_filter
                ) if score <= threshold
            ]
        except RuntimeError:
            vector_candidates = []
        
        # 2) BM25 关键词候选（能精确命中专有名词/编号，弥补向量检索的短板）
        bm25_candidates: List[Tuple[Document, float]] = []
        if self._bm25 is not None:
            metadata_predicate = None
            if metadata_filter:
                filter_items = list(metadata_filter.items())
                metadata_predicate = lambda doc, _items=filter_items: all(
                    doc.metadata.get(key) == value for key, value in _items
                )
            bm25_candidates = self._bm25.search(
                query, k=candidate_k, metadata_predicate=metadata_predicate
            )
        
        # 3) RRF 融合：只依赖排名，规避两个检索器分数尺度不一致的问题
        fused = _rrf_fuse([vector_candidates, bm25_candidates], constant_k=config.HYBRID_FUSION_K)
        return fused[:k]
    
    def get_relevant_context(
        self, 
        query: str, 
        top_k: int = 3,
        score_threshold: float = None,
        metadata_filter: Optional[dict] = None
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
        docs_with_scores = self.similarity_search(
            query,
            k=top_k * 2,
            metadata_filter=metadata_filter
        )
        
        threshold = score_threshold or config.SIMILARITY_THRESHOLD
        
        if config.ENABLE_HYBRID_RETRIEVAL:
            # 混合检索：向量 + BM25，宽召回后 RRF 融合
            docs_with_scores = self.hybrid_search(
                query,
                k=top_k,
                score_threshold=threshold,
                metadata_filter=metadata_filter
            )
        else:
            # 纯向量检索（对比用）：过滤并取前 top_k 个
            docs_with_scores = [
                doc for doc, score in docs_with_scores 
                if score <= threshold
            ][:top_k]
            docs_with_scores = [(doc, 0.0) for doc in docs_with_scores]
        
        if not docs_with_scores:
            return ""
        
        # 合并上下文
        contexts = [doc.page_content for doc, _ in docs_with_scores]
        return "\n\n---\n\n".join(contexts)
    
    def clear(self):
        """清空向量库"""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        self._vectorstore = None
    
    @property
    def vectorstore(self) -> Optional[Chroma]:
        return self._vectorstore
