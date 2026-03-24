"""
FastAPI 后端主应用
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import base64
import json
from typing import Optional

from .models import (
    ChatRequest, ChatResponse, ChatStreamResponse, Source,
    DocumentResponse, AgentChatResponse, AgentStep, AgentAction
)
from ..core import (
    DocumentLoader, DocumentSplitter, VectorStore, RAGChain,
    LLM, RAGAgent
)
from ..core.session import SessionManager
from ..config import config

# 创建 FastAPI 应用
app = FastAPI(
    title="智能文档问答系统 API",
    description="基于 RAG + Agent 的文档问答服务",
    version="2.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化组件
session_manager = SessionManager()


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "智能文档问答系统 API",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


# ========== 会话管理 API ==========

@app.post("/sessions")
async def create_session(name: Optional[str] = None):
    """创建新会话"""
    session = session_manager.create_session(name)
    return session.to_dict()


@app.get("/sessions")
async def list_sessions():
    """列出所有会话"""
    sessions = session_manager.list_sessions()
    return [s.to_dict() for s in sessions]


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """获取会话详情"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session.to_dict()


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    if session_manager.delete_session(session_id):
        return {"message": "会话已删除"}
    raise HTTPException(status_code=404, detail="会话不存在")


@app.post("/sessions/{session_id}/clear")
async def clear_session_messages(session_id: str):
    """清空会话消息"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session_manager.clear_messages(session_id)
    return {"message": "消息已清空"}


# ========== 文档管理 API ==========

@app.post("/sessions/{session_id}/documents")
async def upload_document(session_id: str, file: UploadFile = File(...)):
    """上传文档"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    try:
        # 读取文件内容
        content = await file.read()
        
        # 加载文档
        documents = DocumentLoader.load_from_bytes(content, file.filename)
        
        if not documents:
            raise HTTPException(status_code=400, detail="文档为空或解析失败")
        
        # 文档分块
        splitter = DocumentSplitter()
        splits = splitter.split(documents)
        
        # 创建/加载向量库
        vectorstore = VectorStore(persist_directory=session.vectorstore_path)
        
        # 如果是第一个文档，清空；否则追加
        is_first = len(session.documents) == 0
        vectorstore.create_from_documents(splits, clear_existing=is_first)
        
        # 添加到会话
        session_manager.add_document(session_id, file.filename, len(splits))
        
        return DocumentResponse(
            id=session.documents[-1].id,
            filename=file.filename,
            chunk_count=len(splits)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/documents")
async def list_documents(session_id: str):
    """列出会话的所有文档"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    return [doc.__dict__ for doc in session.documents]


# ========== 聊天 API ==========

@app.post("/chat")
async def chat(request: ChatRequest):
    """普通聊天（非流式）"""
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")
    
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    if not session.documents:
        raise HTTPException(status_code=400, detail="请先上传文档")
    
    try:
        # 加载向量库
        vectorstore = VectorStore(persist_directory=session.vectorstore_path)
        vectorstore.load()
        
        # 创建 RAG Chain
        rag_chain = RAGChain(vectorstore)
        
        # 获取历史
        chat_history = session_manager.get_chat_history(request.session_id)
        
        # 执行 RAG
        response = rag_chain.invoke(request.message, chat_history)
        
        # 保存消息
        session_manager.add_message(
            request.session_id,
            "user",
            request.message
        )
        session_manager.add_message(
            request.session_id,
            "assistant",
            response.answer,
            response.sources
        )
        
        return ChatResponse(
            message=response.answer,
            sources=response.sources,
            is_relevant=response.is_relevant
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天"""
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")
    
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    if not session.documents:
        raise HTTPException(status_code=400, detail="请先上传文档")
    
    async def generate():
        try:
            # 加载向量库
            vectorstore = VectorStore(persist_directory=session.vectorstore_path)
            vectorstore.load()
            
            # 创建 RAG Chain
            rag_chain = RAGChain(vectorstore)
            
            # 获取历史
            chat_history = session_manager.get_chat_history(request.session_id)
            
            # 保存用户消息
            session_manager.add_message(
                request.session_id,
                "user",
                request.message
            )
            
            full_response = ""
            
            # 流式生成
            for chunk in rag_chain.stream(request.message, chat_history):
                full_response += chunk
                data = ChatStreamResponse(delta=chunk, finished=False)
                yield f"data: {json.dumps(data.model_dump(), ensure_ascii=False)}\n\n"
            
            # 获取来源
            response = rag_chain.invoke(request.message, chat_history)
            
            # 保存助手消息
            session_manager.add_message(
                request.session_id,
                "assistant",
                full_response,
                response.sources
            )
            
            # 发送结束标记
            final_data = ChatStreamResponse(
                delta="",
                finished=True,
                sources=[Source(**s) for s in response.sources]
            )
            yield f"data: {json.dumps(final_data.model_dump(), ensure_ascii=False)}\n\n"
            
        except Exception as e:
            error_data = ChatStreamResponse(delta=f"[错误: {str(e)}]", finished=True)
            yield f"data: {json.dumps(error_data.model_dump(), ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


# ========== Agent API ==========

@app.post("/agent/chat")
async def agent_chat(request: ChatRequest):
    """Agent 聊天（支持工具调用）"""
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")
    
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    try:
        # 加载向量库
        vectorstore = VectorStore(persist_directory=session.vectorstore_path)
        vectorstore.load()
        
        # 创建 RAG Agent
        agent = RAGAgent(vectorstore)
        
        # 获取历史
        chat_history = session_manager.get_chat_history(request.session_id)
        
        # 执行 Agent
        response = agent.invoke(request.message, chat_history)
        
        # 转换步骤格式
        thoughts = []
        for step in response.steps:
            thoughts.append(AgentThought(
                thought=step.thought,
                action=AgentAction(
                    tool=step.action or "",
                    input=step.action_input or "",
                    output=step.observation
                ) if step.action else None
            ))
        
        # 保存消息
        session_manager.add_message(
            request.session_id,
            "user",
            request.message
        )
        session_manager.add_message(
            request.session_id,
            "assistant",
            response.final_answer
        )
        
        return AgentChatResponse(
            final_answer=response.final_answer,
            thoughts=thoughts
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 启动命令: uvicorn src.rag_agent.api.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
