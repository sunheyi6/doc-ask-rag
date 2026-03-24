"""
会话管理模块 - 管理多文档和多轮对话
"""
import uuid
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from .vectorstore import VectorStore


@dataclass
class DocumentInfo:
    """文档信息"""
    id: str
    filename: str
    chunk_count: int
    created_at: str
    file_path: Optional[str] = None


@dataclass
class Session:
    """会话"""
    id: str
    name: str
    created_at: str
    updated_at: str
    documents: List[DocumentInfo] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    vectorstore_path: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(**data)


class SessionManager:
    """
    会话管理器
    
    支持多会话、多文档管理
    """
    
    def __init__(self, storage_dir: str = "./data/sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: Dict[str, Session] = {}
        self._load_all_sessions()
    
    def create_session(self, name: str = None) -> Session:
        """创建新会话"""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        session = Session(
            id=session_id,
            name=name or f"会话 {now[:19]}",
            created_at=now,
            updated_at=now,
            vectorstore_path=str(self.storage_dir / session_id / "vectorstore")
        )
        
        self._sessions[session_id] = session
        self._save_session(session)
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        return self._sessions.get(session_id)
    
    def list_sessions(self) -> List[Session]:
        """列出所有会话"""
        return sorted(
            self._sessions.values(),
            key=lambda s: s.updated_at,
            reverse=True
        )
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self._sessions:
            # 删除存储文件
            session_file = self._get_session_file(session_id)
            if session_file.exists():
                session_file.unlink()
            
            # 删除向量库
            session = self._sessions[session_id]
            if session.vectorstore_path:
                import shutil
                shutil.rmtree(session.vectorstore_path, ignore_errors=True)
            
            del self._sessions[session_id]
            return True
        return False
    
    def add_message(self, session_id: str, role: str, content: str, sources: List = None):
        """添加消息到会话"""
        session = self._sessions.get(session_id)
        if not session:
            return
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        if sources:
            message["sources"] = sources
        
        session.messages.append(message)
        session.updated_at = datetime.now().isoformat()
        
        self._save_session(session)
    
    def add_document(self, session_id: str, filename: str, chunk_count: int, file_path: str = None):
        """添加文档到会话"""
        session = self._sessions.get(session_id)
        if not session:
            return
        
        doc_info = DocumentInfo(
            id=str(uuid.uuid4()),
            filename=filename,
            chunk_count=chunk_count,
            created_at=datetime.now().isoformat(),
            file_path=file_path
        )
        
        session.documents.append(doc_info)
        session.updated_at = datetime.now().isoformat()
        
        self._save_session(session)
    
    def get_chat_history(self, session_id: str, limit: int = 10) -> str:
        """获取格式化的聊天历史"""
        session = self._sessions.get(session_id)
        if not session:
            return ""
        
        messages = session.messages[-limit:] if len(session.messages) > limit else session.messages
        
        history_parts = []
        for msg in messages:
            role = "用户" if msg["role"] == "user" else "助手"
            history_parts.append(f"{role}: {msg['content']}")
        
        return "\n".join(history_parts)
    
    def clear_messages(self, session_id: str):
        """清空会话消息"""
        session = self._sessions.get(session_id)
        if session:
            session.messages = []
            session.updated_at = datetime.now().isoformat()
            self._save_session(session)
    
    def _get_session_file(self, session_id: str) -> Path:
        """获取会话文件路径"""
        return self.storage_dir / f"{session_id}.json"
    
    def _save_session(self, session: Session):
        """保存会话到文件"""
        session_file = self._get_session_file(session.id)
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
    
    def _load_all_sessions(self):
        """加载所有会话"""
        if not self.storage_dir.exists():
            return
        
        for session_file in self.storage_dir.glob("*.json"):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    session = Session.from_dict(data)
                    self._sessions[session.id] = session
            except Exception as e:
                print(f"加载会话失败 {session_file}: {e}")


# 全局会话管理器实例
session_manager = SessionManager()
