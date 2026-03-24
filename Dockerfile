# 智能文档问答系统 Dockerfile
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY .env.example ./.env.example

# 创建数据目录
RUN mkdir -p /app/data/sessions

# 设置环境变量
ENV PYTHONPATH=/app/src
ENV PORT=8000

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "uvicorn", "rag_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
