# AGENTS.md - 智能文档问答系统

> 本文档供 AI 编程助手阅读，帮助理解本项目架构和开发规范。

## 项目概述

这是一个基于 RAG（检索增强生成）技术的**智能文档问答系统**，用户可上传 PDF、TXT、DOCX 文档，系统基于文档内容回答用户问题。

### 核心功能
- 📄 支持 PDF、TXT、DOCX 格式文档上传
- 🧠 基于向量检索的智能问答
- 💬 支持多轮对话和流式输出
- 📖 引用溯源 - 显示回答来源段落
- 🔍 无关问题智能拦截（避免回答与文档无关的问题）
- 🤖 ReAct Agent 架构 - 支持工具调用和复杂推理
- 🌐 使用阿里云通义千问大模型和 Embedding 服务

## 技术栈

| 组件 | 技术 |
|------|------|
| 编程语言 | Python 3.10+ |
| Web 框架 | Streamlit |
| API 框架 | FastAPI |
| Agent 框架 | LangGraph + ReAct |
| RAG 框架 | LangChain |
| 向量数据库 | ChromaDB |
| 大语言模型 | 阿里云通义千问 (qwen-turbo, qwen-plus) |
| Embedding | DashScope text-embedding-v1 |
| 文档加载 | PyPDFLoader, TextLoader, Docx2txtLoader |

## 项目结构

```
doc-ask-rag/
├── src/rag_agent/          # 核心源码
│   ├── __init__.py
│   ├── config.py           # 集中配置管理
│   ├── data/sample_docs/   # 内置示例文档（首次启动自动导入知识库）
│   ├── core/               # 核心模块
│   │   ├── __init__.py
│   │   ├── agent.py        # ReAct Agent 实现
│   │   ├── document_loader.py  # 文档加载器
│   │   ├── embeddings.py   # Embedding 服务
│   │   ├── llm.py          # 大模型封装
│   │   ├── rag_chain.py    # RAG 流程链
│   │   ├── session.py      # 会话管理
│   │   ├── text_splitter.py    # 文本分块
│   │   ├── tools.py        # Agent 工具
│   │   └── vectorstore.py  # 向量数据库操作
│   ├── api/                # FastAPI 后端
│   │   ├── __init__.py
│   │   ├── main.py         # FastAPI 主应用
│   │   └── models.py       # Pydantic 模型
│   ├── web/                # Streamlit Web 界面
│   │   ├── __init__.py
│   │   └── app.py          # Streamlit 主应用
│   └── utils/              # 工具函数
│       ├── __init__.py
│       └── logger.py       # 日志记录
├── scripts/                # 启动脚本
│   ├── run_web.py          # 启动 Web 界面
│   └── run_api.py          # 启动 API 服务
├── tests/                  # 测试文件
├── docs/                   # 文档
├── chroma_db/              # 持久化向量数据库
├── chroma_db_temp/         # 临时向量库
├── uploaded_docs/          # 上传的文档存储
├── logs/                   # 日志文件
├── .env                    # 环境变量（API 密钥）
├── .env.example            # 环境变量模板
├── requirements.txt        # Python 依赖
├── Dockerfile              # Docker 配置
├── docker-compose.yml      # Docker Compose 配置
└── AGENTS.md               # 本文件
```

## 模块详解

### 1. src/rag_agent/config.py

**集中配置管理**，使用 dataclass 和单例模式：

```python
@dataclass
class Config:
    DASHSCOPE_API_KEY: str
    LLM_MODEL: str = "qwen-turbo"
    EMBEDDING_MODEL: str = "text-embedding-v1"
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100
    SIMILARITY_THRESHOLD: float = 1.8
    RETRIEVAL_TOP_K: int = 5
    # ...

config = Config.from_env()
```

### 2. src/rag_agent/core/

**核心功能模块**：

- **agent.py** - ReAct Agent 实现，支持工具调用（计算、搜索等）
- **document_loader.py** - 文档加载，支持 PDF/TXT/DOCX
- **text_splitter.py** - 文本分块，针对中文标点优化
- **embeddings.py** - DashScope Embedding 封装
- **vectorstore.py** - ChromaDB 向量数据库操作
- **llm.py** - 通义千问大模型封装，支持流式输出
- **rag_chain.py** - RAG 流程链，整合检索和生成
- **session.py** - 多用户会话管理
- **tools.py** - Agent 可用工具（计算器、搜索等）

### 3. src/rag_agent/web/app.py

**Streamlit Web 应用**：

- 文件上传与处理
- 实时流式输出（打字机效果）
- 引用溯源展示
- 多轮对话历史
- 无关问题拦截

### 4. src/rag_agent/api/

**FastAPI 后端**（可选）：

- RESTful API 接口
- WebSocket 支持流式输出
- 多文档管理
- 会话管理

## 开发规范

### 代码风格

1. **注释规范**：使用中文注释，模块顶部说明功能
2. **代码结构**：按功能分区，使用清晰的函数和类
3. **分隔符**：使用 `# === 标题 ===` 或 `# ==========` 划分代码区域
4. **成功标记**：使用 `✅` emoji 标记成功状态
5. **错误处理**：使用 `try-except` 包裹文件操作和 API 调用
6. **类型注解**：函数参数和返回值使用类型注解
7. **日志记录**：使用 `src/rag_agent/utils/logger.py` 而非 print

### 环境变量

必须在 `.env` 文件中配置：
```bash
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
```

可选配置：
```bash
# 模型配置
LLM_MODEL=qwen-turbo
EMBEDDING_MODEL=text-embedding-v1

# 文本分块
CHUNK_SIZE=600
CHUNK_OVERLAP=100

# 检索配置
RETRIEVAL_TOP_K=5
SIMILARITY_THRESHOLD=1.8

# 路径配置
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_TEMP_DIR=./chroma_db_temp
UPLOAD_DIR=./uploaded_docs

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

**安全注意**：`.env` 文件已加入 `.gitignore`，**切勿将 API 密钥提交到 Git**。

### 依赖管理

使用 `requirements.txt` 管理依赖：
```bash
# 安装依赖
pip install -r requirements.txt
```

主要依赖：
- streamlit
- fastapi / uvicorn
- langchain / langgraph
- dashscope（阿里云大模型 SDK）
- chromadb
- pypdf / python-docx

## 运行和测试

### 启动 Web 应用（推荐）

**使用启动脚本（推荐）：**
```bash
# Windows
venv\Scripts\python scripts/run_web.py

# macOS/Linux
venv/bin/python scripts/run_web.py
```

**或者直接运行 Streamlit：**
```bash
venv\Scripts\python -m streamlit run src/rag_agent/web/app.py
```

访问：http://localhost:8501

### 启动 API 服务

```bash
venv\Scripts\python scripts/run_api.py
```

访问：http://localhost:8000
API 文档：http://localhost:8000/docs

### Docker 部署

```bash
# 构建并启动
docker-compose up -d

# 只构建 Web 服务
docker-compose up -d web

# 查看日志
docker-compose logs -f
```

## 向量数据库

- **chroma_db/**: 持久化存储，保留历史文档
- **chroma_db_temp/**: 临时存储，Web 应用默认使用，重启后清空
- 距离度量：余弦距离（cosine）
- 相似度分数范围：0~2（越小越相似）

## 无关问题拦截策略

系统通过多层策略防止回答与文档无关的问题：

1. **相似度阈值**：检索最相关文档片段，余弦距离 > SIMILARITY_THRESHOLD 视为无关
2. **LLM 约束**：提示词明确要求"如果上下文没有相关信息，请回答'您提问的问题与文档无关'"
3. **Agent 判断**：ReAct Agent 可主动判断问题相关性

## Git 提交规范

提交格式：
```
feat: 添加 xxx 功能
fix: 修复 xxx 问题
docs: 更新文档
refactor: 重构 xxx 模块
test: 添加测试
chore: 构建/工具变更
```

## 开发环境注意事项

1. **⚠️ 不要自动停止运行的项目**：后台任务会在超时后自动停止，不适合开发服务器
   - 开发时请直接在终端中运行命令，保持窗口打开
   - 停止服务：按 `Ctrl+C` 或在终端中关闭

2. **虚拟环境使用**：
   ```bash
   # Windows - 使用虚拟环境 Python
   venv\Scripts\python scripts/run_web.py
   
   # 不要直接使用 python，可能调用系统 Python
   # python scripts/run_web.py  ❌ 不推荐
   ```

## 安全注意事项

1. ⚠️ **API 密钥管理**：永远不要在代码中硬编码 API 密钥，使用环境变量
2. ⚠️ **文件上传**：上传的文件存储在 `uploaded_docs/`，定期清理
3. ⚠️ **日志文件**：日志可能包含敏感信息，注意保护 `logs/` 目录

## 已废弃文件

以下旧文件已废弃，仅作历史参考：
- `app.py` - 旧版 Streamlit 应用
- `rag_pipeline.py` - 旧版 RAG 管道
- `rag_conversation.py` - 旧版对话 RAG
- `test_llm.py` - 旧版测试脚本

**新项目请使用 `scripts/run_web.py` 启动。**

## 协作记忆规则（用户口头约定）

当用户明确提出“记一下 / 以后按这个来”的长期规则时：

1. 若属于项目级开发或运行约束，写入 `AGENTS.md` 持久化。
2. 启动项目时优先使用 `AGENTS.md` 约定方式（当前为 `venv\Scripts\python scripts/run_web.py`，默认端口 `8501`）。
3. 重启服务时必须先关闭旧进程，再启动新进程，避免端口冲突与多实例并存。
4. 若新约束与旧约束冲突，以用户最新明确指令为准，并同步更新本文件。
5. **消融实验约束**：完成设计或实现后，做消融实验——尝试移除本次涉及的不必要抽象和设计，以原目标和验收标准验证；仍满足就删，不满足就恢复。（已同步至全局用户约束 AGENTS.md）
