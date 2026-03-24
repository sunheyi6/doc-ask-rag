# 📄 智能文档问答系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://langchain.com/)

> 🤖 基于 RAG（检索增强生成）技术的个人知识库助手，支持文档问答、引用溯源、流式输出

## ✨ 核心特性

- 📚 **多格式支持** - PDF、TXT、DOCX 文档一键上传
- 🔍 **智能检索** - 基于向量相似度的语义检索
- 💬 **流式对话** - 实时打字机效果输出
- 📖 **引用溯源** - 显示回答来源段落
- 🛡️ **无关问题拦截** - 智能识别与文档无关的问题
- 🧠 **多轮对话** - 支持上下文理解的连续对话

## 🚀 快速开始

### 环境要求

- Python 3.10+
- 阿里云 DashScope API Key

### 安装

```bash
# 克隆项目
git clone <your-repo-url>
cd doc-ask-rag

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 配置

创建 `.env` 文件：

```bash
# 复制模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key
DASHSCOPE_API_KEY=sk-your-api-key-here
```

### 运行

```bash
# 方式1：直接运行
python -m streamlit run src/rag_agent/web/app.py

# 方式2：使用脚本
python scripts/run_web.py
```

访问 http://localhost:8501

## 📸 功能演示

### 文档上传与问答

1. 在侧边栏上传文档（PDF/TXT/DOCX）
2. 系统自动解析、分块、建立向量索引
3. 在对话框中输入问题
4. 查看带有引用来源的回答

### 引用溯源

每个回答都可以展开查看引用来源，显示相关段落和相似度分数。

## 🏗️ 架构设计

```
doc-ask-rag/
├── src/rag_agent/          # 核心源码
│   ├── core/               # 核心模块
│   │   ├── document_loader.py    # 文档加载
│   │   ├── text_splitter.py      # 文本分块
│   │   ├── embeddings.py         # Embedding 服务
│   │   ├── vectorstore.py        # 向量数据库
│   │   ├── llm.py                # 大模型封装
│   │   └── rag_chain.py          # RAG 流程
│   ├── api/                # API 接口（待实现）
│   ├── web/                # Web 界面
│   │   └── app.py          # Streamlit 应用
│   └── utils/              # 工具函数
├── tests/                  # 测试
├── docs/                   # 文档
├── scripts/                # 脚本
├── requirements.txt        # 依赖
└── README.md               # 本文件
```

## 🔧 技术栈

| 组件 | 技术 |
|------|------|
| Web 框架 | Streamlit |
| RAG 框架 | LangChain |
| 向量数据库 | ChromaDB |
| 大语言模型 | 阿里云通义千问 |
| Embedding | DashScope text-embedding-v1 |
| 文档解析 | PyPDF2, python-docx |

## 📖 进阶配置

### 环境变量

```bash
# 模型配置
LLM_MODEL=qwen-turbo              # 大模型
EMBEDDING_MODEL=text-embedding-v1 # Embedding 模型

# 文本分块
CHUNK_SIZE=600                    # 分块大小
CHUNK_OVERLAP=100                 # 重叠大小

# 检索配置
RETRIEVAL_TOP_K=5                 # 检索结果数量
SIMILARITY_THRESHOLD=1.8          # 相似度阈值

# 路径配置
CHROMA_PERSIST_DIR=./chroma_db    # 向量库存储路径
UPLOAD_DIR=./uploaded_docs        # 上传文件路径
```

## 🛣️ 路线图

### 阶段 1：基础完善 ✅
- [x] 项目结构重构
- [x] 流式输出
- [x] 引用溯源
- [ ] README 完善

### 阶段 2：Agent 能力 🚧
- [ ] FastAPI 后端
- [ ] ReAct Agent 架构
- [ ] 工具调用（计算、搜索）
- [ ] 多文档管理

### 阶段 3：工程化
- [ ] Docker 部署
- [ ] 测试覆盖
- [ ] 线上部署

## 🤝 贡献

欢迎提交 Issue 和 PR！

## 📄 License

MIT License

---

> 💡 提示：本项目使用阿里云 DashScope 服务，请确保你的账户有足够的额度。
