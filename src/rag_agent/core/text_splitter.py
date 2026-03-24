"""
文本分割模块 - 封装文档分块逻辑
"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..config import config


class DocumentSplitter:
    """文档分割器"""
    
    # 针对中文优化的分隔符
    SEPARATORS = ["\n\n", "\n", "。", "？", "！", " ", ""]
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.SEPARATORS,
            length_function=len,
            is_separator_regex=False
        )
    
    def split(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        return self._splitter.split_documents(documents)
    
    def split_text(self, text: str) -> List[str]:
        """分割纯文本"""
        return self._splitter.split_text(text)
