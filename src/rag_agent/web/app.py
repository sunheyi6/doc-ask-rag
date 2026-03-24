"""
Streamlit Web 应用 - 个人知识库问答系统
支持：多会话管理、多文档知识库、引用溯源、流式输出
"""
import sys
import os
import time
import gc
import shutil
import hashlib
import uuid
import re
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

from rag_agent.config import config
from rag_agent.core import (
    DocumentLoader,
    DocumentSplitter,
    VectorStore,
    RAGChain,
)
from rag_agent.utils.logger import conversation_logger


# ========== 页面配置 ==========
st.set_page_config(
    page_title="个人知识库问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ========== 状态初始化 ==========
if "app_session_id" not in st.session_state:
    st.session_state.app_session_id = str(uuid.uuid4())
    conversation_logger.logger.info(f"应用启动: {st.session_state.app_session_id}")

if "show_sources" not in st.session_state:
    st.session_state.show_sources = True

if "active_rag_chain" not in st.session_state:
    st.session_state.active_rag_chain = None
if "active_rag_chain_vector_dir" not in st.session_state:
    st.session_state.active_rag_chain_vector_dir = None
if "kb_current_files" not in st.session_state:
    st.session_state.kb_current_files = []
if "kb_active_filename" not in st.session_state:
    st.session_state.kb_active_filename = None
if "kb_current_vector_dir" not in st.session_state:
    st.session_state.kb_current_vector_dir = None
if "kb_uploaded_signature" not in st.session_state:
    st.session_state.kb_uploaded_signature = None
if "kb_files_meta" not in st.session_state:
    st.session_state.kb_files_meta = []
if "last_auto_import_signature" not in st.session_state:
    st.session_state.last_auto_import_signature = None


def _create_chat(title: Optional[str] = None) -> Dict[str, Any]:
    """创建会话对象"""
    return {
        "id": str(uuid.uuid4()),
        "title": title or "新对话",
        "messages": [],
    }


if "chats" not in st.session_state:
    st.session_state.chats = [_create_chat("新对话 1")]
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = st.session_state.chats[0]["id"]


def _safe_cleanup_vector_dir(path: str, max_retries: int = 5, wait_seconds: float = 0.2) -> None:
    """安全清理旧向量目录（Windows 文件锁时重试，失败不阻塞主流程）"""
    if not path or not os.path.exists(path):
        return
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path, ignore_errors=False)
            return
        except Exception as e:
            if attempt == max_retries - 1:
                conversation_logger.log_error(e, f"清理旧向量目录失败: {path}")
                return
            time.sleep(wait_seconds)


def _build_temp_vector_dir(chat_id: str) -> str:
    """为会话生成独立临时向量目录"""
    return os.path.join(config.CHROMA_TEMP_DIR, st.session_state.app_session_id, chat_id, str(uuid.uuid4()))


def _build_upload_signature(uploaded_files: list) -> str:
    """根据上传文件内容生成签名，用于判断是否需要重建知识库"""
    digest = hashlib.md5()
    for f in sorted(uploaded_files, key=lambda x: x.name):
        content = f.getvalue()
        digest.update(f.name.encode("utf-8"))
        digest.update(str(len(content)).encode("utf-8"))
        digest.update(content)
    return digest.hexdigest()


def _build_registry_signature(files_meta: List[Dict[str, str]]) -> str:
    """根据知识库文件注册表生成签名"""
    digest = hashlib.md5()
    for item in sorted(files_meta, key=lambda x: x["name"]):
        path = item["path"]
        size = os.path.getsize(path) if os.path.exists(path) else 0
        digest.update(item["name"].encode("utf-8"))
        digest.update(str(size).encode("utf-8"))
    return digest.hexdigest()


def _save_uploaded_file(uploaded_file) -> Dict[str, str]:
    """保存上传文件到 UPLOAD_DIR，返回文件元信息"""
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    safe_name = uploaded_file.name
    saved_name = f"{uuid.uuid4().hex}_{safe_name}"
    saved_path = os.path.join(config.UPLOAD_DIR, saved_name)
    with open(saved_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return {"name": safe_name, "path": saved_path}


def _rebuild_shared_kb(files_meta: List[Dict[str, str]]) -> Optional[RAGChain]:
    """
    根据知识库文件注册表重建共享向量库。
    返回可用的 RAGChain；若无文件则返回 None。
    """
    old_vector_dir = st.session_state.kb_current_vector_dir
    st.session_state.active_rag_chain = None
    st.session_state.active_rag_chain_vector_dir = None
    gc.collect()

    if not files_meta:
        st.session_state.kb_current_files = []
        st.session_state.kb_active_filename = None
        st.session_state.kb_current_vector_dir = None
        st.session_state.kb_uploaded_signature = None
        _safe_cleanup_vector_dir(old_vector_dir)
        return None

    all_documents = []
    for item in files_meta:
        docs = DocumentLoader.load(item["path"])
        if not docs:
            continue
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["filename"] = item["name"]
        all_documents.extend(docs)

    if not all_documents:
        st.session_state.kb_current_files = []
        st.session_state.kb_active_filename = None
        st.session_state.kb_current_vector_dir = None
        st.session_state.kb_uploaded_signature = None
        _safe_cleanup_vector_dir(old_vector_dir)
        return None

    splitter = DocumentSplitter()
    splits = splitter.split(all_documents)

    new_vector_dir = _build_temp_vector_dir("shared")
    os.makedirs(new_vector_dir, exist_ok=True)
    vectorstore = VectorStore(persist_directory=new_vector_dir)
    vectorstore.create_from_documents(splits, clear_existing=False)

    st.session_state.kb_current_files = [item["name"] for item in files_meta]
    if st.session_state.kb_active_filename not in st.session_state.kb_current_files:
        st.session_state.kb_active_filename = st.session_state.kb_current_files[0]
    st.session_state.kb_current_vector_dir = new_vector_dir
    st.session_state.kb_uploaded_signature = _build_registry_signature(files_meta)

    st.session_state.active_rag_chain = RAGChain(vectorstore)
    st.session_state.active_rag_chain_vector_dir = new_vector_dir

    _safe_cleanup_vector_dir(old_vector_dir)
    return st.session_state.active_rag_chain


def _is_uploaded_files_query(question: str) -> bool:
    """判断是否在询问已上传文件列表"""
    q = re.sub(r"\s+", "", (question or "").strip().lower())
    patterns = [
        r"我(刚才|刚刚|现在)?上传了(什么|哪些|哪几个)?文件",
        r"(现在|当前)?(有|上传了)?(哪些|什么|多少)文档",
        r"(知识库|资料库)(里|中)?(目前|现在|当前)?(有)?(几个|多少个|多少)(文件|文档)",
        r"(知识库|资料库)(里|中)?(文件|文档)(有)?(多少|几个)",
        r"(我的|当前)(文件|文档)(列表|有哪些|是什么)",
        r"(你|系统)知道我上传了什么",
    ]
    return any(re.search(p, q) for p in patterns)


def _get_current_chat() -> Dict[str, Any]:
    """获取当前会话对象"""
    for chat in st.session_state.chats:
        if chat["id"] == st.session_state.current_chat_id:
            return chat
    st.session_state.current_chat_id = st.session_state.chats[0]["id"]
    return st.session_state.chats[0]


def _build_chat_title_from_messages(messages: List[Dict[str, str]]) -> str:
    """根据前几条用户问题自动生成会话标题"""
    user_questions = [m.get("content", "").strip() for m in messages if m.get("role") == "user" and m.get("content")]
    if not user_questions:
        return "新对话"

    seed = " / ".join(user_questions[:2])
    return seed[:18] + ("..." if len(seed) > 18 else "")


def _ensure_rag_chain() -> Optional[RAGChain]:
    """确保当前应用有可用的共享知识库 RAGChain（按需加载）"""
    if not st.session_state.kb_current_vector_dir:
        st.session_state.active_rag_chain = None
        st.session_state.active_rag_chain_vector_dir = None
        return None

    if (
        st.session_state.active_rag_chain is not None
        and st.session_state.active_rag_chain_vector_dir == st.session_state.kb_current_vector_dir
    ):
        return st.session_state.active_rag_chain

    vectorstore = VectorStore(persist_directory=st.session_state.kb_current_vector_dir)
    loaded = vectorstore.load()
    if loaded is None:
        return None

    chain = RAGChain(vectorstore)
    st.session_state.active_rag_chain = chain
    st.session_state.active_rag_chain_vector_dir = st.session_state.kb_current_vector_dir
    return chain


# ========== 左侧栏：仅会话管理 ==========
with st.sidebar:
    st.title("💬 会话")

    if st.button("➕ 新建对话", use_container_width=True):
        new_index = len(st.session_state.chats) + 1
        new_chat = _create_chat(f"新对话 {new_index}")
        st.session_state.chats.insert(0, new_chat)
        st.session_state.current_chat_id = new_chat["id"]
        st.rerun()

    st.caption("对话列表")
    for chat in st.session_state.chats:
        is_current = chat["id"] == st.session_state.current_chat_id
        if st.button(
            f"🟢 {chat['title']}" if is_current else chat["title"],
            key=f"chat_tab_{chat['id']}",
            use_container_width=True,
            type="primary" if is_current else "secondary",
        ):
            if not is_current:
                st.session_state.current_chat_id = chat["id"]
                st.rerun()

    st.divider()
    st.subheader("⚙️ 设置")
    st.session_state.show_sources = st.toggle("显示引用来源", value=st.session_state.show_sources)
    if st.session_state.kb_current_files:
        if st.session_state.kb_active_filename not in st.session_state.kb_current_files:
            st.session_state.kb_active_filename = st.session_state.kb_current_files[0]
        st.session_state.kb_active_filename = st.selectbox(
            "当前检索文件",
            options=st.session_state.kb_current_files,
            index=st.session_state.kb_current_files.index(st.session_state.kb_active_filename),
            help="问答与引用来源只会使用该文件",
        )
    else:
        st.caption("当前无可选文件")

    st.divider()
    st.subheader("ℹ️ 系统信息")
    st.caption(f"LLM: {config.LLM_MODEL}")
    st.caption(f"Embedding: {config.EMBEDDING_MODEL}")
    st.caption(f"检索阈值: {config.SIMILARITY_THRESHOLD}")


current_chat = _get_current_chat()
rag_chain = _ensure_rag_chain()


st.title("📚 个人知识库问答")


# ========== 文件上传区域 ==========
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_files = st.file_uploader(
        "📤 上传文档（可多选，自动导入知识库）",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="支持 PDF、TXT、DOCX 格式；选择后会自动导入并重建索引",
        key="uploader_shared_kb",
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.kb_current_files:
        st.success(f"✅ 已加载文档：{len(st.session_state.kb_current_files)} 个（全部会话共享）")
        st.caption("、".join(st.session_state.kb_current_files[:3]) + ("..." if len(st.session_state.kb_current_files) > 3 else ""))
    else:
        st.info("⬆️ 请上传一个或多个文档")


# ========== 处理上传（自动导入） ==========
if uploaded_files:
    upload_signature = _build_upload_signature(uploaded_files)
    should_auto_import = upload_signature != st.session_state.last_auto_import_signature
else:
    should_auto_import = False

if should_auto_import and uploaded_files:
    with st.spinner(f"正在导入并重建知识库（{len(uploaded_files)} 个文件）..."):
        try:
            config.validate()

            existing = {item["name"]: item for item in st.session_state.kb_files_meta}
            added_count = 0
            replaced_count = 0

            for f in uploaded_files:
                new_item = _save_uploaded_file(f)
                if new_item["name"] in existing:
                    old_path = existing[new_item["name"]]["path"]
                    if os.path.exists(old_path):
                        try:
                            os.remove(old_path)
                        except Exception as e:
                            conversation_logger.log_error(e, f"删除旧上传文件失败: {old_path}")
                    existing[new_item["name"]] = new_item
                    replaced_count += 1
                else:
                    existing[new_item["name"]] = new_item
                    added_count += 1

            st.session_state.kb_files_meta = list(existing.values())
            rag_chain = _rebuild_shared_kb(st.session_state.kb_files_meta)
            st.session_state.last_auto_import_signature = upload_signature

            for filename in st.session_state.kb_current_files:
                conversation_logger.log_document_upload(
                    session_id=st.session_state.app_session_id,
                    filename=filename,
                    chunk_count=0,
                )

            st.success(
                f"🎉 导入完成：新增 {added_count}，替换同名 {replaced_count}。"
                f" 当前知识库共 {len(st.session_state.kb_current_files)} 个文档。"
            )
        except Exception as e:
            st.error(f"❌ 处理失败：{str(e)}")
            conversation_logger.log_error(e, f"文档导入失败 - 会话: {current_chat['id']}")
            st.stop()

# 兜底：某些重跑场景下，前面阶段可能拿不到 rag_chain，这里再尝试一次
if rag_chain is None and st.session_state.kb_current_vector_dir:
    rag_chain = _ensure_rag_chain()

# ========== 知识库管理 ==========
with st.expander("🗂 知识库管理", expanded=False):
    if not st.session_state.kb_files_meta:
        st.caption("当前知识库为空。请先上传并导入文档。")
    else:
        doc_rows = []
        for item in st.session_state.kb_files_meta:
            size_kb = round((os.path.getsize(item["path"]) / 1024), 1) if os.path.exists(item["path"]) else 0
            doc_rows.append({"文件名": item["name"], "大小(KB)": size_kb})
        st.dataframe(doc_rows, use_container_width=True, hide_index=True)

        selected_doc = st.selectbox(
            "选择要删除的文档",
            options=[item["name"] for item in st.session_state.kb_files_meta],
            key="kb_manage_selected_doc",
        )

        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("🗑 删除选中文档", use_container_width=True):
                keep_items = []
                removed_path = None
                for item in st.session_state.kb_files_meta:
                    if item["name"] == selected_doc and removed_path is None:
                        removed_path = item["path"]
                        continue
                    keep_items.append(item)

                if removed_path and os.path.exists(removed_path):
                    try:
                        os.remove(removed_path)
                    except Exception as e:
                        conversation_logger.log_error(e, f"删除知识库文件失败: {removed_path}")

                st.session_state.kb_files_meta = keep_items
                rag_chain = _rebuild_shared_kb(st.session_state.kb_files_meta)
                st.success(f"已删除文档：{selected_doc}")
                st.rerun()

        with action_col2:
            if st.button("🔄 重建索引", use_container_width=True):
                rag_chain = _rebuild_shared_kb(st.session_state.kb_files_meta)
                st.success("知识库索引已重建")


# ========== 聊天记录 ==========
st.divider()
for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        is_identity = msg.get("is_identity_question", False)
        sources_list = msg.get("sources") or []
        has_sources = len(sources_list) > 0
        is_irrelevant = "与文档无关" in msg.get("content", "")
        is_meta_style_answer = msg.get("content", "").strip().startswith("我是小A，您的个人知识库助手")

        if msg["role"] == "assistant" and st.session_state.show_sources and has_sources and not is_identity and not is_irrelevant and not is_meta_style_answer:
            with st.expander("📚 查看引用来源"):
                file_best_score = {}
                for source in sources_list:
                    source_file = source.get("metadata", {}).get("filename", "未知文件")
                    score = source.get("score", 999)
                    if source_file not in file_best_score or score < file_best_score[source_file]:
                        file_best_score[source_file] = score

                for i, (source_file, best_score) in enumerate(file_best_score.items(), 1):
                    st.markdown(f"来源 {i} | 文件: {source_file} | 相关度: {best_score}")


# ========== 用户输入 ==========
if rag_chain is None:
    st.warning("👆 请先上传一个或多个文档以开始问答")
else:
    prompt = st.chat_input("请输入您的问题...")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)

        current_chat["messages"].append({"role": "user", "content": prompt})
        current_chat["title"] = _build_chat_title_from_messages(current_chat["messages"])

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            start_time = time.time()

            with st.spinner("思考中..."):
                try:
                    # 会话内系统问题：直接回答已上传文件列表，不走文档检索
                    if _is_uploaded_files_query(prompt):
                        if st.session_state.kb_current_files:
                            file_list = "\n".join([f"{idx + 1}. {name}" for idx, name in enumerate(st.session_state.kb_current_files)])
                            full_response = f"您当前已上传 {len(st.session_state.kb_current_files)} 个文件：\n{file_list}"
                        else:
                            full_response = "当前还没有上传文件。"
                        message_placeholder.write(full_response)
                        sources = []
                        is_identity = True
                    else:
                        response = rag_chain.invoke(
                            prompt,
                            target_filename=st.session_state.kb_active_filename,
                        )

                        if not response.is_relevant:
                            full_response = response.answer
                            message_placeholder.write(full_response)
                            sources = []
                            is_identity = True
                        elif response.is_identity_question:
                            full_response = response.answer
                            message_placeholder.write(full_response)
                            sources = []
                            is_identity = True
                        else:
                            for chunk in rag_chain.stream(
                                prompt,
                                target_filename=st.session_state.kb_active_filename,
                            ):
                                full_response += chunk
                                message_placeholder.write(full_response + "▌")
                            message_placeholder.write(full_response)
                            sources = response.sources
                            is_identity = False

                    response_time = time.time() - start_time

                    conversation_logger.log_conversation(
                        session_id=current_chat["id"],
                        user_message=prompt,
                        ai_response=full_response,
                        sources=sources,
                        is_relevant=True if _is_uploaded_files_query(prompt) else response.is_relevant,
                        response_time=response_time,
                    )

                    current_chat["messages"].append(
                        {
                            "role": "assistant",
                            "content": full_response,
                            "sources": sources,
                            "is_identity_question": is_identity,
                        }
                    )
                except Exception as e:
                    st.error(f"❌ 生成失败：{str(e)}")
                    conversation_logger.log_error(e, f"对话生成失败 - 会话: {current_chat['id']}")
