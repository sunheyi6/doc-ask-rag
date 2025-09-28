# app.py - 最终完整版：智能文档问答系统（支持自定义无关提示）
import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_community.llms import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import tempfile
import shutil

# ========== 1. 加载环境变量 ==========
load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    if "DASHSCOPE_API_KEY" in st.secrets:
        api_key = st.secrets["DASHSCOPE_API_KEY"]
    else:
        st.error("❌ 未找到 DASHSCOPE_API_KEY！请在 .env 文件中设置，或使用 Streamlit Secrets。")
        st.stop()

# ========== 2. 初始化 Session State ==========
st.title("📄 智能文档问答系统")
st.caption("上传文件，基于内容智能回答")

UPLOAD_DIR = "./uploaded_docs"
CHROMA_DIR = "./chroma_db_temp"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ========== 3. 文件上传与处理 ==========
uploaded_file = st.file_uploader("📤 上传 PDF/TXT/DOCX 文件", type=["pdf", "txt", "docx"])

if uploaded_file is not None:
    # 判断是否为新文件
    if st.session_state.current_file != uploaded_file.name:
        with st.spinner(f"正在处理 {uploaded_file.name} ..."):

            # 清理旧向量库（确保重建）
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR)
            os.makedirs(CHROMA_DIR, exist_ok=True)

            # 使用临时目录保存文件
            with tempfile.TemporaryDirectory() as tmp_dir:
                file_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                try:
                    # === 加载文档 ===
                    if uploaded_file.name.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                    elif uploaded_file.name.endswith(".txt"):
                        encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312"]
                        data = None
                        for enc in encodings:
                            try:
                                loader = TextLoader(file_path, encoding=enc)
                                data = loader.load()
                                st.success(f"✅ 使用编码 {enc} 成功读取 TXT")
                                break
                            except:
                                continue
                        if data is None:
                            st.error("❌ 所有编码均无法读取该 TXT 文件，请检查格式。")
                            st.stop()
                    elif uploaded_file.name.endswith(".docx"):
                        loader = Docx2txtLoader(file_path)
                    else:
                        st.error("❌ 不支持的文件格式")
                        st.stop()

                    documents = loader.load()
                    if not documents or len(documents) == 0:
                        st.error("❌ 文档为空或解析失败，请检查文件内容。")
                        st.stop()

                    # 显示文档前200字
                    st.write("📄 **文档内容预览（前200字）:**")
                    st.text(documents[0].page_content[:200])

                    # === 分割文本（优化分块策略）===
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=600,
                        chunk_overlap=100,
                        separators=["\n\n", "\n", "。", "？", "！", " ", ""]
                    )
                    docs = text_splitter.split_documents(documents)
                    st.write(f"🧠 已分割为 {len(docs)} 个文本块。")

                    # === 创建向量库（强制使用 cosine 距离）===
                    embedding = DashScopeEmbeddings(
                        dashscope_api_key=api_key,
                        model="text-embedding-v1"
                    )

                    vectorstore = Chroma.from_documents(
                        docs,
                        embedding,
                        persist_directory=CHROMA_DIR,
                        collection_metadata={"hnsw:space": "cosine"}  # 使用余弦距离
                    )
                    vectorstore.persist()

                    st.session_state.vectorstore = vectorstore
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.messages = []  # 清空历史
                    st.success("🎉 文档处理完成，可以开始提问！")

                except Exception as e:
                    st.error(f"❌ 处理失败：{str(e)}")
                    st.stop()

# ========== 4. 聊天界面 ==========
if st.session_state.vectorstore is None:
    st.info("请先上传文件以开始问答。")
else:
    st.write(f"📌 当前文档：**{st.session_state.current_file}**")

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 用户输入
    if prompt := st.chat_input("请输入问题"):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    # === 智能拦截数学常识问题 ===
                    is_math_question = bool(re.search(r'\d+\s*[+\-*/]\s*\d+', prompt))
                    short_question = len(prompt.strip()) < 20

                    if is_math_question and short_question:
                        response = "您提问的问题与文档无关，请提问与文档相关问题"
                        st.write(response)
                    else:
                        # === 执行检索（取 top 3，选最相关的一个）===
                        docs_with_scores = st.session_state.vectorstore.similarity_search_with_score(prompt, k=3)
                        
                        if not docs_with_scores:
                            response = "您提问的问题与文档无关，请提问与文档相关问题"
                            st.write(response)
                        else:
                            # 取距离最小（最相关）的结果
                            best_doc, best_score = min(docs_with_scores, key=lambda x: x[1])
                            
                            # 显示调试信息
                            st.write(f"🔍 相似度 (越小越相关，0~2): {best_score:.3f}")
                            st.text(f"相关段落: {best_doc.page_content[:300]}...")

                            # ✅ 放宽阈值：> 1.5 才认为不相关
                            if best_score > 1.5:
                                response = "您提问的问题与文档无关，请提问与文档相关问题"
                                st.write(response)
                            else:
                                # 调用 LLM 回答
                                llm = Tongyi(model="qwen-turbo", dashscope_api_key=api_key)
                                final_prompt = f"""请根据以下上下文回答问题。如果上下文没有相关信息，请回答"您提问的问题与文档无关，请提问与文档相关问题"。不要编造。

上下文：
{best_doc.page_content}

问题：
{prompt}

回答："""
                                response = llm.invoke(final_prompt)
                                st.write(response)

                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"❌ 回答失败：{str(e)}")
                    error_msg = "抱歉，系统出错。"
                    st.write(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})