"""
配置文件 - 集中管理所有配置项
"""
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

# 加载环境变量
load_dotenv()


@dataclass
class Config:
    """应用配置类"""
    
    # API Keys
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    
    # 模型配置
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen-turbo")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
    
    # 文本分块配置
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "600"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    
    # 向量数据库配置
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_TEMP_DIR: str = os.getenv("CHROMA_TEMP_DIR", "./chroma_db_temp")
    
    # 相似度阈值（余弦距离，越小越相似）
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "1.8"))
    
    # 检索配置
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    
    # 上传文件配置
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploaded_docs")
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "50"))  # MB
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/app.log")
    
    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置"""
        return cls()
    
    def validate(self) -> bool:
        """验证配置是否有效"""
        if not self.DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY 未设置，请在 .env 文件中配置")
        return True


# 全局配置实例
config = Config.from_env()
