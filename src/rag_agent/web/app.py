"""
Streamlit Web 应用 - 智能文档问答系统（重构版）
支持：流式输出、引用溯源、多文档管理
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import tempfile
from pathlib import Path

from rag_agent.config import config
from rag_agent.core import (
    DocumentLoader,
    DocumentSplitter,
    VectorStore,
    RAGChain
)
from rag_agent.utils.logger import conversation_logger

# 生成或获取会话ID
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
    conversation_logger.logger.info(f"新会话启动: {st.session_state.session_id}")

# ========== 页面配置 ==========
st.set_page_config(
    page_title="智能文档问答系统",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 初始化 Session State ==========
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_sources" not in st.session_state:
    st.session_state.show_sources = True

# ========== 侧边栏 ==========
with st.sidebar:
    st.title("📄 智能文档问答")
    st.caption("基于 RAG 技术的个人知识库助手")
    
    st.divider()
    
    # 设置选项
    st.subheader("⚙️ 设置")
    st.session_state.show_sources = st.toggle("显示引用来源", value=True)
    
    # 清空对话
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # 关于
    st.subheader("ℹ️ 关于")
    st.markdown("""
    **小A** - 您的个人知识库助手
    
    支持格式：PDF、TXT、DOCX
    
    技术栈：
    - LangChain
    - ChromaDB
    - 通义千问大模型
    """)

# ========== 主界面 ==========
st.title("💬 智能问答")

# 文件上传区域
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📤 上传文档",
        type=["pdf", "txt", "docx"],
        help="支持 PDF、TXT、DOCX 格式"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.current_file:
        st.success(f"✅ 当前文档：{st.session_state.current_file}")
    else:
        st.info("⬆️ 请先上传文档")

# 处理上传的文件
if uploaded_file is not None:
    if st.session_state.current_file != uploaded_file.name:
        with st.spinner(f"正在处理 {uploaded_file.name}..."):
            try:
                # 验证配置
                config.validate()
                
                # 从字节流加载文档
                documents = DocumentLoader.load_from_bytes(
                    uploaded_file.getvalue(),
                    uploaded_file.name
                )
                
                if not documents:
                    st.error("❌ 文档为空或解析失败")
                    st.stop()
                
                # 显示预览
                preview = documents[0].page_content[:500]
                with st.expander("📄 文档预览（前500字）"):
                    st.text(preview)
                
                # 文档分块
                splitter = DocumentSplitter()
                splits = splitter.split(documents)
                
                st.info(f"📝 文档已分割为 {len(splits)} 个片段")
                
                # 创建向量库
                vectorstore = VectorStore(persist_directory=config.CHROMA_TEMP_DIR)
                vectorstore.create_from_documents(splits, clear_existing=True)
                
                # 创建 RAG Chain
                st.session_state.rag_chain = RAGChain(vectorstore)
                st.session_state.current_file = uploaded_file.name
                st.session_state.messages = []
                
                # 记录文档上传日志
                conversation_logger.log_document_upload(
                    session_id=st.session_state.session_id,
                    filename=uploaded_file.name,
                    chunk_count=len(splits)
                )
                
                st.success("🎉 文档处理完成，可以开始提问！")
                
            except Exception as e:
                st.error(f"❌ 处理失败：{str(e)}")
                conversation_logger.log_error(e, f"文档上传失败 - 会话: {st.session_state.session_id}")
                st.stop()

# 聊天界面
st.divider()

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # 显示引用来源（身份问题不显示，空来源也不显示）
        is_identity = msg.get("is_identity_question", False)
        has_sources = msg.get("sources") and len(msg["sources"]) > 0
        if msg["role"] == "assistant" and st.session_state.show_sources and has_sources and not is_identity:
            with st.expander("📚 查看引用来源"):
                for i, source in enumerate(msg["sources"], 1):
                    st.markdown(f"**来源 {i}** (相关度: {source['score']})")
                    st.text(source["content"][:500])
                    st.divider()

# 用户输入
if st.session_state.rag_chain is None:
    st.warning("👆 请先上传文档以开始问答")
else:
    if prompt := st.chat_input("请输入您的问题..."):
        # 显示用户消息
        with st.chat_message("user"):
            st.write(prompt)
        
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # 生成回答
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            start_time = time.time()
            
            # 流式生成
            with st.spinner("思考中..."):
                try:
                    # 先调用一次获取响应类型和来源（不重复调用 LLM 生成）
                    response = st.session_state.rag_chain.invoke(prompt)
                    
                    if not response.is_relevant:
                        # 不相关问题，直接显示，不显示引用来源
                        full_response = response.answer
                        message_placeholder.write(full_response)
                        sources = []
                        is_identity = True  # 设为 True 以避免显示引用来源
                    elif response.is_identity_question:
                        # 元问题（身份/能力），直接显示，不流式
                        full_response = response.answer
                        message_placeholder.write(full_response)
                        sources = []
                        is_identity = True
                    else:
                        # 文档相关问题，流式输出
                        full_response = ""
                        for chunk in st.session_state.rag_chain.stream(prompt):
                            full_response += chunk
                            message_placeholder.write(full_response + "▌")
                        message_placeholder.write(full_response)
                        
                        # 使用 invoke 获取的来源（避免 stream 重复检索）
                        sources = response.sources
                        is_identity = False
                    
                    # 计算响应时间
                    response_time = time.time() - start_time
                    
                    # 记录对话日志
                    conversation_logger.log_conversation(
                        session_id=st.session_state.session_id,
                        user_message=prompt,
                        ai_response=full_response,
                        sources=sources,
                        is_relevant=response.is_relevant,
                        response_time=response_time
                    )
                    
                    # 保存到历史
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources,
                        "is_identity_question": is_identity
                    })
                    
                except Exception as e:
                    st.error(f"❌ 生成失败：{str(e)}")
                    conversation_logger.log_error(e, f"对话生成失败 - 会话: {st.session_state.session_id}")
