# 📚 个人知识库问答系统（Doc Ask RAG）

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://langchain.com/)

基于 RAG（检索增强生成）的个人知识库系统，支持多会话问答、共享知识库、多文档上传、单文件检索约束、引用溯源与自动化评测。

## ✨ 当前能力

- 多格式上传：`PDF / TXT / DOCX`
- 多会话管理：左侧会话页签，当前会话高亮
- 共享知识库：上传文档在所有会话中共享
- 单文件检索：可指定“当前检索文件”
- 流式输出：打字机效果
- 引用来源展示：按文件去重，仅显示`来源序号 | 文件名 | 相关度`
- 无关问题拦截：对常识类/非文档问题进行拒答
- 元问题识别：支持口语问法（如“你是谁呀”）
- 系统状态问答：支持“我上传了什么文件”等非检索型系统问题
- 自动化评测：支持生成评测集并输出 benchmark 报告

## 🧱 项目结构

```text
doc-ask-rag/
├── src/rag_agent/
│   ├── core/
│   │   ├── document_loader.py
│   │   ├── text_splitter.py
│   │   ├── embeddings.py
│   │   ├── vectorstore.py
│   │   ├── llm.py
│   │   └── rag_chain.py
│   ├── web/
│   │   └── app.py
│   ├── api/
│   └── utils/
├── scripts/
│   ├── run_web.py
│   ├── run_api.py
│   ├── evaluate_rag.py
│   └── generate_eval_dataset.py
├── docs/
│   ├── EVALUATION.md
│   ├── benchmark.md
│   ├── benchmark_real.md
│   └── eval_dataset_*.json
├── tests/
├── uploaded_docs/
├── requirements.txt
└── AGENTS.md
```

## 🚀 快速开始

### 1) 安装依赖

```bash
# Windows
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
```

### 2) 配置环境变量

创建 `.env` 文件（可参考 `.env.example`）：

```bash
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 3) 启动 Web（推荐）

```bash
# 按项目规范启动（默认端口 8501）
venv\Scripts\python scripts/run_web.py
```

访问：[http://localhost:8501](http://localhost:8501)

## 🧭 使用说明

1. 上传一个或多个文档（构建共享知识库）
2. 在左侧切换/新建会话（会话只隔离聊天历史，不隔离知识库）
3. 在左侧设置中选择：
   - 是否显示引用来源
   - 当前检索文件（限制回答只基于该文件）
4. 输入问题并查看流式回答

## 📊 自动化评测

### 1) 用现有评测集执行评测

```bash
venv\Scripts\python scripts/evaluate_rag.py ^
  --docs uploaded_docs/resume.pdf ^
  --dataset docs/eval_dataset_resume_30.json ^
  --output docs/benchmark.md
```

### 2) 基于真实文档自动生成评测集（30条）

```bash
venv\Scripts\python scripts/generate_eval_dataset.py ^
  --docs "uploaded_docs/阿里巴巴Java开发手册(终极版).pdf" ^
  --output docs/eval_dataset_java_spec_30.json
```

### 3) 对生成的评测集执行评测

```bash
venv\Scripts\python scripts/evaluate_rag.py ^
  --docs "uploaded_docs/阿里巴巴Java开发手册(终极版).pdf" ^
  --dataset docs/eval_dataset_java_spec_30.json ^
  --output docs/benchmark_java_spec.md
```

评测指标定义见：[EVALUATION.md](./docs/EVALUATION.md)

## 🔧 关键配置

```bash
# 模型
LLM_MODEL=qwen-turbo
EMBEDDING_MODEL=text-embedding-v1

# 检索
RETRIEVAL_TOP_K=5
SIMILARITY_THRESHOLD=1.8

# 文本分块
CHUNK_SIZE=600
CHUNK_OVERLAP=100

# 路径
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_TEMP_DIR=./chroma_db_temp
UPLOAD_DIR=./uploaded_docs
```

## 🧪 测试

```bash
venv\Scripts\python -m pytest -q
```

## 📌 开发约定

- 启动优先使用：`venv\Scripts\python scripts/run_web.py`
- 默认端口：`8501`
- 重启服务必须先关闭旧进程再启动新进程
- 项目级协作约束统一维护在 [AGENTS.md](./AGENTS.md)

## 📄 License

MIT
