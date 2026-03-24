"""
RAG Chain 模块 - 整合检索和生成的完整流程
"""
from typing import Iterator, Optional, Dict, Any, List
from dataclasses import dataclass
from langchain_core.documents import Document
from .vectorstore import VectorStore
from .llm import LLM, RAGPromptBuilder
from ..config import config
import re


@dataclass
class RAGResponse:
    """RAG 响应结果"""
    answer: str
    sources: List[Dict[str, Any]]  # 引用来源
    is_relevant: bool  # 是否与文档相关
    

class RAGChain:
    """RAG 问答链"""
    
    def __init__(self, vectorstore: VectorStore = None):
        self.vectorstore = vectorstore or VectorStore()
        self.llm = LLM()
    
    def _is_math_question(self, question: str) -> bool:
        """检测是否为简单数学问题"""
        # 检测数学表达式如 "1+1", "2*3" 等
        pattern = r'^\s*\d+\s*[+\-*/]\s*\d+\s*$'
        return bool(re.match(pattern, question.strip()))
    
    def _check_relevance(self, question: str) -> tuple[bool, str]:
        """
        检查问题是否与文档相关
        
        Returns:
            (is_relevant, context)
        """
        # 简单数学问题直接判定为不相关
        if self._is_math_question(question) and len(question.strip()) < 20:
            return False, ""
        
        # 获取相关上下文
        context = self.vectorstore.get_relevant_context(
            question,
            top_k=3,
            score_threshold=config.SIMILARITY_THRESHOLD
        )
        
        # 如果获取不到上下文，判定为不相关
        if not context:
            return False, ""
        
        return True, context
    
    def invoke(self, question: str, chat_history: str = "") -> RAGResponse:
        """
        执行 RAG 问答（同步）
        
        Args:
            question: 用户问题
            chat_history: 历史对话
            
        Returns:
            RAGResponse 包含答案和引用
        """
        # 检查相关性
        is_relevant, context = self._check_relevance(question)
        
        if not is_relevant:
            return RAGResponse(
                answer="您提问的问题与文档无关，请提问与文档相关问题",
                sources=[],
                is_relevant=False
            )
        
        # 构建提示词
        prompt = RAGPromptBuilder.build(context, question, chat_history)
        
        # 调用 LLM
        answer = self.llm.invoke(prompt)
        
        # 获取来源信息
        sources = self._get_sources(question)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            is_relevant=True
        )
    
    def stream(self, question: str, chat_history: str = "") -> Iterator[str]:
        """
        流式执行 RAG 问答
        
        Args:
            question: 用户问题
            chat_history: 历史对话
            
        Yields:
            生成的文本片段
        """
        # 检查相关性
        is_relevant, context = self._check_relevance(question)
        
        if not is_relevant:
            yield "您提问的问题与文档无关，请提问与文档相关问题"
            return
        
        # 构建提示词
        prompt = RAGPromptBuilder.build(context, question, chat_history)
        
        # 流式调用 LLM
        for chunk in self.llm.stream(prompt):
            yield chunk
    
    def _get_sources(self, question: str, k: int = 3) -> List[Dict[str, Any]]:
        """获取引用来源"""
        docs_with_scores = self.vectorstore.similarity_search(question, k=k)
        
        sources = []
        for doc, score in docs_with_scores[:k]:
            if score <= config.SIMILARITY_THRESHOLD:
                sources.append({
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "score": round(score, 3),
                    "metadata": doc.metadata
                })
        
        return sources
