"""
API 数据模型 - Pydantic 模型定义
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    filename: str = Field(..., description="文件名")
    content: str = Field(..., description="Base64 编码的文件内容")


class DocumentResponse(BaseModel):
    """文档响应"""
    id: str = Field(..., description="文档ID")
    filename: str = Field(..., description="文件名")
    chunk_count: int = Field(..., description="分块数量")
    created_at: datetime = Field(default_factory=datetime.now)


class ChatMessage(BaseModel):
    """聊天消息"""
    role: str = Field(..., description="角色: user/assistant")
    content: str = Field(..., description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str = Field(..., description="用户消息")
    session_id: Optional[str] = Field(None, description="会话ID")
    stream: bool = Field(False, description="是否流式输出")


class ChatResponse(BaseModel):
    """聊天响应"""
    message: str = Field(..., description="助手回复")
    sources: List[Dict[str, Any]] = Field(default=[], description="引用来源")
    is_relevant: bool = Field(True, description="是否与文档相关")


class Source(BaseModel):
    """引用来源"""
    content: str = Field(..., description="来源内容")
    score: float = Field(..., description="相似度分数")
    metadata: Dict[str, Any] = Field(default={}, description="元数据")


class ChatStreamResponse(BaseModel):
    """流式聊天响应"""
    delta: str = Field(..., description="新增文本")
    finished: bool = Field(False, description="是否结束")
    sources: Optional[List[Source]] = Field(None, description="引用来源")


class AgentAction(BaseModel):
    """Agent 动作"""
    tool: str = Field(..., description="工具名称")
    input: str = Field(..., description="工具输入")
    output: Optional[str] = Field(None, description="工具输出")
    

class AgentThought(BaseModel):
    """Agent 思考过程"""
    thought: str = Field(..., description="思考内容")
    action: Optional[AgentAction] = Field(None, description="执行的动作")


class AgentChatResponse(BaseModel):
    """Agent 聊天响应"""
    final_answer: str = Field(..., description="最终回答")
    thoughts: List[AgentThought] = Field(default=[], description="思考过程")
    sources: List[Source] = Field(default=[], description="引用来源")
