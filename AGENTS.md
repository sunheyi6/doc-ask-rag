# AGENTS.md - 智能文档问答系统

> 本文档供 AI 编程助手阅读，帮助理解本项目架构和开发规范。

## 项目概述

这是一个基于 RAG（检索增强生成）技术的**智能文档问答系统**，用户可上传 PDF、TXT、DOCX 文档，系统基于文档内容回答用户问题。

### 核心功能
- 📄 支持 PDF、TXT、DOCX 格式文档上传
- 🧠 基于向量检索的智能问答
- 💬 支持多轮对话（ rag_conversation.py ）
- 🔍 无关问题智能拦截（避免回答与文档无关的问题）
- 🌐 使用阿里云通义千问大模型和 Embedding 服务

## 技术栈

| 组件 | 技术 |
|------|------|
| 编程语言 | Python 3.x |
| Web 框架 | Streamlit |
| RAG 框架 | LangChain (langchain-community, langchain-core, langchain-text-splitters) |
| 向量数据库 | ChromaDB |
| 大语言模型 | 阿里云通义千问 (Tongyi/qwen-turbo, qwen-plus) |
| Embedding | DashScope text-embedding-v1 |
| 文档加载 | PyPDFLoader, TextLoader, Docx2txtLoader |

## 项目结构

```
doc-ask-rag/
├── app.py                  # 主应用：Streamlit Web 界面（最完整版本）
├── rag_pipeline.py         # 基础 RAG 管道：PDF → 向量库 → 问答
├── rag_conversation.py     # 对话版 RAG：支持连续对话和记忆
├── test_llm.py             # LLM 连接测试脚本
├── requirements.txt        # Python 依赖
├── .env                    # 环境变量（API 密钥）
├── .env.example            # 环境变量模板
├── chroma_db/              # 持久化向量数据库（默认）
├── chroma_db_temp/         # 临时向量库（app.py 使用）
├── uploaded_docs/          # 上传的文档存储
└── AGENTS.md               # 本文件
```

## 模块详解

### 1. app.py（主应用）

**最完整的 Streamlit Web 应用版本**，包含：

- 文件上传与处理（PDF/TXT/DOCX）
- 文本分块（chunk_size=600, chunk_overlap=100）
- 向量库构建（使用 cosine 距离）
- 聊天界面与历史记录
- 无关问题拦截逻辑：
  - 数学公式检测（`\d+\s*[+\-*/]\s*\d+`）
  - 相似度阈值判断（score > 1.5 视为无关）
  - LLM 提示词约束

**关键配置：**
```python
CHROMA_DIR = "./chroma_db_temp"
chunk_size = 600
chunk_overlap = 100
model = "qwen-turbo"  # 轻量级模型，响应更快
similarity_threshold = 1.5  # 余弦距离阈值
```

### 2. rag_pipeline.py

**基础 RAG 管道**，用于离线处理单个 PDF：

- 加载 PDF → 文本分块 → 向量化 → 构建问答链
- 使用自定义 `DashScopeEmbeddings` 类包装 dashscope SDK
- 支持自定义 Prompt 模板
- 默认处理 `resume.pdf`

**关键配置：**
```python
chunk_size = 1000
chunk_overlap = 200
model = "qwen-plus"  # 更强的模型
persist_directory = "./chroma_db"
```

### 3. rag_conversation.py

**支持记忆的对话版 RAG**：

- 使用 `ConversationBufferMemory` 保存对话历史
- 使用 `ConversationalRetrievalChain` 实现多轮对话
- 命令行交互模式（输入 'quit' 或 'exit' 退出）

### 4. test_llm.py

**最简单的测试脚本**，用于验证 LLM API 连接是否正常。

## 开发规范

### 代码风格

1. **注释规范**：使用中文注释，模块顶部说明功能
2. **代码结构**：按功能分区（加载 → 分块 → 向量化 → 问答）
3. **分隔符**：使用 `# === 标题 ===` 或 `# ==========` 划分代码区域
4. **成功标记**：使用 `✅` emoji 标记成功状态
5. **错误处理**：使用 `try-except` 包裹文件操作和 API 调用

### 环境变量

必须在 `.env` 文件中配置：
```bash
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
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
- langchain-community / langchain-core / langchain-text-splitters
- dashscope（阿里云大模型 SDK）
- chromadb
- pypdf / python-docx（文档解析）

## 运行和测试

### 启动 Web 应用

**开发环境运行方式（推荐）：**

在开发环境中，建议使用前台运行方式，保持终端窗口打开：

```bash
# 使用虚拟环境运行
venv\Scripts\python -m streamlit run app.py

# 或者使用 streamlit 命令
venv\Scripts\streamlit run app.py
```

**注意事项：**
- ⚠️ **不要依赖后台任务运行项目** - 使用 `run_in_background=true` 启动的后台任务会在超时后自动停止（默认60秒），不适合长时间运行的开发服务器
- 开发时请直接在终端中运行上述命令，保持窗口打开
- 服务启动后访问：http://localhost:8501

**生产环境部署：**
```bash
streamlit run app.py --server.headless true
```

### 运行基础 RAG 管道
```bash
python rag_pipeline.py
```

### 运行对话版 RAG
```bash
python rag_conversation.py
```

### 测试 LLM 连接
```bash
python test_llm.py
```

## 向量数据库

- **chroma_db/**: `rag_pipeline.py` 和 `rag_conversation.py` 使用的持久化存储
- **chroma_db_temp/**: `app.py` 使用的临时存储，每次上传新文件会清空重建
- 距离度量：余弦距离（cosine），相似度分数范围 0~2（越小越相似）

## 无关问题拦截策略

系统通过多层策略防止回答与文档无关的问题：

1. **数学问题拦截**：检测到数学表达式（如 `1+1`）且问题较短时直接拒绝
2. **相似度阈值**：检索最相关文档片段，余弦距离 > 1.5 视为无关
3. **LLM 约束**：提示词明确要求“如果上下文没有相关信息，请回答‘您提问的问题与文档无关，请提问与文档相关问题’”

## 已知限制

1. **单文档模式**：`app.py` 一次只支持处理一个文档，上传新文档会清空历史
2. **中文优化**：文本分割器针对中文标点（。、？、！）优化
3. **编码处理**：TXT 文件支持 UTF-8、GBK、GB2312 编码自动检测
4. **临时文件**：使用 `tempfile.TemporaryDirectory` 处理上传文件，不永久保存原始文件

## Git 提交规范

当前项目仅有一个提交：`first commit`

建议后续提交使用以下格式：
```
feat: 添加 xxx 功能
fix: 修复 xxx 问题
docs: 更新文档
refactor: 重构 xxx 模块
```

## 开发环境注意事项

1. **⚠️ 不要自动停止运行的项目**：后台任务 (`run_in_background=true`) 会在超时后自动停止，不适合开发服务器
   - Streamlit 开发服务器需要保持持续运行
   - 在开发环境中，直接在终端前台运行命令，保持窗口打开
   - 停止服务：按 `Ctrl+C` 或在终端中关闭

2. **虚拟环境激活**：
   ```bash
   # Windows
   venv\Scripts\python -m streamlit run app.py
   ```

## 安全注意事项

1. ⚠️ **API 密钥管理**：永远不要在代码中硬编码 API 密钥，使用环境变量
2. ⚠️ **文件上传**：上传的文件临时存储在内存和临时目录，定期清理 `chroma_db_temp/`
3. ⚠️ **无关问题拦截**：当前阈值（1.5）可能需要根据实际文档调整
