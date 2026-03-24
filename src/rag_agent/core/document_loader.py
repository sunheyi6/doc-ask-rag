"""
文档加载模块 - 支持多种格式
"""
import os
import tempfile
from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)


class DocumentLoader:
    """文档加载器 - 支持 PDF、TXT、DOCX"""
    
    # 支持的文件格式
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}
    
    # TXT 文件编码尝试列表
    ENCODINGS = ["utf-8", "utf-8-sig", "gbk", "gb2312"]
    
    @classmethod
    def load(cls, file_path: str) -> List[Document]:
        """
        加载文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            Document 列表
            
        Raises:
            ValueError: 不支持的文件格式
            RuntimeError: 文件读取失败
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {ext}，支持的格式: {cls.SUPPORTED_EXTENSIONS}")
        
        try:
            if ext == ".pdf":
                return cls._load_pdf(file_path)
            elif ext == ".txt":
                return cls._load_txt(file_path)
            elif ext == ".docx":
                return cls._load_docx(file_path)
        except Exception as e:
            raise RuntimeError(f"加载文件失败: {e}")
    
    @classmethod
    def _load_pdf(cls, file_path: str) -> List[Document]:
        """加载 PDF 文件"""
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    @classmethod
    def _load_txt(cls, file_path: str) -> List[Document]:
        """加载 TXT 文件，自动检测编码"""
        for encoding in cls.ENCODINGS:
            try:
                loader = TextLoader(file_path, encoding=encoding)
                return loader.load()
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"无法解码文件，已尝试编码: {cls.ENCODINGS}")
    
    @classmethod
    def _load_docx(cls, file_path: str) -> List[Document]:
        """加载 DOCX 文件"""
        loader = Docx2txtLoader(file_path)
        return loader.load()
    
    @classmethod
    def load_from_bytes(cls, file_bytes: bytes, filename: str) -> List[Document]:
        """
        从字节流加载文档
        
        Args:
            file_bytes: 文件内容字节
            filename: 文件名（用于确定格式）
            
        Returns:
            Document 列表
        """
        ext = Path(filename).suffix.lower()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, filename)
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            return cls.load(file_path)
