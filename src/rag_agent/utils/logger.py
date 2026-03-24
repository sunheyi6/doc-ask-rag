"""
日志模块 - 记录用户对话和系统日志
"""
import os
import sys
from datetime import datetime
from pathlib import Path
from loguru import logger as _logger
from ..config import config


class ConversationLogger:
    """对话日志记录器"""
    
    def __init__(self, log_dir: str = None):
        self.log_dir = Path(log_dir or "./logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 对话日志文件
        self.conversation_file = self.log_dir / "conversations.log"
        
        # 系统日志文件
        self.system_file = self.log_dir / "system.log"
        
        # 配置 loguru
        self._setup_logger()
        
        # 暴露 logger 属性
        self.logger = _logger
    
    def _setup_logger(self):
        """配置日志记录器"""
        # 移除默认处理器
        _logger.remove()
        
        # 添加控制台输出
        _logger.add(
            sys.stdout,
            level=config.LOG_LEVEL,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True
        )
        
        # 添加系统日志文件
        _logger.add(
            self.system_file,
            level=config.LOG_LEVEL,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="30 days",
            encoding="utf-8"
        )
    
    def log_conversation(self, session_id: str, user_message: str, ai_response: str, 
                         sources: list = None, is_relevant: bool = True, 
                         response_time: float = None):
        """
        记录对话日志
        
        Args:
            session_id: 会话ID
            user_message: 用户消息
            ai_response: AI回答
            sources: 引用来源
            is_relevant: 是否与文档相关
            response_time: 响应时间（秒）
        """
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "session_id": session_id,
            "user_message": user_message,
            "ai_response": ai_response[:500] + "..." if len(ai_response) > 500 else ai_response,
            "sources_count": len(sources) if sources else 0,
            "is_relevant": is_relevant,
            "response_time": f"{response_time:.2f}s" if response_time else None
        }
        
        # 写入对话日志文件
        with open(self.conversation_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"时间: {timestamp}\n")
            f.write(f"会话: {session_id}\n")
            f.write(f"用户: {user_message}\n")
            f.write(f"AI: {ai_response[:1000]}...\n" if len(ai_response) > 1000 else f"AI: {ai_response}\n")
            f.write(f"来源数: {log_entry['sources_count']}, 相关: {is_relevant}")
            if response_time:
                f.write(f", 响应时间: {log_entry['response_time']}")
            f.write("\n")
        
        # 同时记录到系统日志
        _logger.info(f"对话记录 | 会话: {session_id[:8]}... | 用户: {user_message[:50]}... | "
                    f"相关: {is_relevant} | 来源: {log_entry['sources_count']}")
    
    def log_error(self, error: Exception, context: str = ""):
        """记录错误日志"""
        _logger.error(f"错误: {context} - {str(error)}")
    
    def log_document_upload(self, session_id: str, filename: str, chunk_count: int):
        """记录文档上传日志"""
        _logger.info(f"文档上传 | 会话: {session_id[:8]}... | 文件: {filename} | 分块: {chunk_count}")
        
        with open(self.conversation_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"[文档上传] 时间: {datetime.now().isoformat()}\n")
            f.write(f"会话: {session_id}\n")
            f.write(f"文件: {filename}\n")
            f.write(f"分块数: {chunk_count}\n")


# 全局日志实例
conversation_logger = ConversationLogger()

# 导出 logger 供其他模块使用
logger = _logger
