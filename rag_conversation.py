# rag_chat.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_community.llms import Tongyi
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import dashscope


# ==================== 自定义 Embedding ====================
class DashScopeEmbeddings(Embeddings):
    def __init__(self, model="text-embedding-v1", api_key=None):
        self.model = model
        if api_key:
            dashscope.api_key = api_key

    def embed_documents(self, texts):
        responses = []
        for text in texts:
            response = dashscope.TextEmbedding.call(model=self.model, input=text)
            if response.status_code == 200:
                responses.append(response.output['embeddings'][0]['embedding'])
            else:
                raise RuntimeError(f"Embedding failed: {response.message}")
        return responses

    def embed_query(self, text):
        return self.embed_documents([text])[0]


# 加载环境变量
load_dotenv()

# ==================== 1. 加载 PDF ====================
loader = PyPDFLoader("resume.pdf")
docs = loader.load()
print(f"✅ 文档加载成功，共 {len(docs)} 页")

# ==================== 2. 分块 ====================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"✅ 文档已分块，共 {len(splits)} 个文本块")

# ==================== 3. 向量化 ====================
embedding = DashScopeEmbeddings(
    model="text-embedding-v1",
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory="./chroma_db"
)
print("✅ 向量数据库构建完成")

# ==================== 4. 大模型 ====================
llm = Tongyi(
    model="qwen-plus",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# ==================== 5. 记忆模块 ====================
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ==================== 6. Prompt ====================
prompt_template = """请根据以下上下文和聊天历史回答问题。
如果无法从上下文中找到答案，请回答“我不知道”。不要编造答案。

上下文：
{context}

聊天历史：
{chat_history}

问题：
{question}

回答："""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])

# ==================== 7. 创建对话链（关键：关闭 verbose）====================
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT},
    verbose=False  # ✅ 关闭调试日志
)
print("✅ 支持连续对话的 RAG 系统已启动！\n")
print("🎙️ 已进入连续对话模式，输入 'quit' 或 'exit' 退出")

while True:
    question = input("\n👤 你：").strip()
    
    if question.lower() in ["quit", "exit", "退出"]:
        print("👋 再见！")
        break
    
    if not question:
        continue  # 忽略空输入
    
    try:
        result = qa_chain.invoke({"question": question})
        print(f"🤖 AI：{result['answer']}")
    except Exception as e:
        print(f"❌ 错误：{e}")
print