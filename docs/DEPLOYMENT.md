# 部署指南

## 本地部署

### 使用 Docker Compose（推荐）

1. 确保已安装 Docker 和 Docker Compose

2. 配置环境变量：
```bash
cp .env.example .env
# 编辑 .env 文件，填入 DASHSCOPE_API_KEY
```

3. 启动服务：
```bash
docker-compose up -d
```

4. 访问 API：
- API 地址: http://localhost:8000
- API 文档: http://localhost:8000/docs

### 使用 Docker

```bash
# 构建镜像
docker build -t rag-agent .

# 运行容器
docker run -d \
  -p 8000:8000 \
  -e DASHSCOPE_API_KEY=your-api-key \
  -v $(pwd)/data:/app/data \
  --name rag-agent \
  rag-agent
```

## 云服务器部署

### 阿里云 ECS 部署

1. 购买 ECS 实例（建议 2核4G 以上）

2. 安装 Docker：
```bash
curl -fsSL https://get.docker.com | sh
systemctl enable docker
systemctl start docker
```

3. 上传项目代码：
```bash
scp -r . root@your-server-ip:/opt/rag-agent
```

4. 在服务器上启动：
```bash
cd /opt/rag-agent
docker-compose up -d
```

5. 配置安全组，开放 8000 端口

### 使用 Nginx 反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## 监控与日志

### 查看日志

```bash
# Docker Compose
docker-compose logs -f api

# Docker
docker logs -f rag-agent
```

### 健康检查

```bash
curl http://localhost:8000/health
```

## 备份与恢复

### 备份数据

```bash
# 备份会话数据
tar -czf backup-$(date +%Y%m%d).tar.gz data/
```

### 恢复数据

```bash
tar -xzf backup-20240101.tar.gz
```
