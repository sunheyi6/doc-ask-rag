# rag_pipeline.py
# 使用 dashscope SDK 手动实现 Embedding，兼容 LangChain

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Tongyi
from langchain.chains import RetrievalQA
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import dashscope  # ✅ 直接使用 dashscope SDK


# ==================== 自定义 Embedding 类 ====================
class DashScopeEmbeddings(Embeddings):
    """使用 dashscope SDK 调用通义千问 Embedding 模型"""

    def __init__(self, model="text-embedding-v1", api_key=None):
        self.model = model
        if api_key:
            dashscope.api_key = api_key  # 设置 API Key

    def embed_documents(self, texts):
        """对文档列表进行向量化"""
        responses = []
        for text in texts:
            response = dashscope.TextEmbedding.call(
                model=self.model,
                input=text
            )
            if response.status_code == 200:
                responses.append(response.output['embeddings'][0]['embedding'])
            else:
                raise RuntimeError(f"Embedding failed: {response.message}")
        return responses

    def embed_query(self, text):
        """对单个查询进行向量化"""
        return self.embed_documents([text])[0]


# 加载环境变量
load_dotenv()

# ==================== 1. 加载 PDF 文档 ====================
# ⚠️ 请将 'your_file.pdf' 改成你 docs 目录下的实际文件名
loader = PyPDFLoader("resume.pdf")  # ✅ 修改为你的实际文件名
docs = loader.load()
print(f"✅ 文档加载成功，共 {len(docs)} 页")
# 在 docs = loader.load() 之后添加
print("\n📄 实际解析出的文本预览：")
print(docs[0].page_content[:1000])  # 打印前 1000 字符

# ==================== 2. 分块（Chunking） ====================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
splits = text_splitter.split_documents(docs)
print(f"✅ 文档已分块，共 {len(splits)} 个文本块")

# ==================== 3. 向量化 + 存入 Chroma ====================
# 使用自定义的 DashScopeEmbeddings
embedding = DashScopeEmbeddings(
    model="text-embedding-v1",
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 创建向量数据库
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory="./chroma_db"
)
print("✅ 向量数据库构建完成，已保存到 ./chroma_db")

# ==================== 4. 创建大模型实例（用于生成回答） ====================
llm = Tongyi(
    model="qwen-plus",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# ==================== 5. 创建 RAG 链（带 Prompt 优化） ====================
prompt_template = """请严格根据以下上下文内容回答问题。只使用上下文中的信息，不要添加任何外部知识或猜测。

如果上下文中没有明确答案，请回答“我不知道”。

上下文：
{context}

问题：
{question}

请用中文回答，并尽量使用原文中的关键词和结构。
回答："""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

print("✅ RAG 问答链已准备就绪！")

# ==================== 6. 测试提问 ====================
def ask_question(question):
    result = qa_chain.invoke({"query": question})
    print(f"\n❓ 问题：{question}")
    print(f"💡 回答：{result['result']}")

# 测试问题
ask_question("这份文档讲了什么？")
ask_question("作者有哪些工作经历？")
ask_question("掌握了哪些技术栈？")