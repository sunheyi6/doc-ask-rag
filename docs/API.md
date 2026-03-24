# API 文档

## 基础信息

- 基础 URL: `http://localhost:8000`
- 文档地址: `http://localhost:8000/docs`

## 会话管理

### 创建会话

```http
POST /sessions
Content-Type: application/json

{
  "name": "我的会话"
}
```

响应：
```json
{
  "id": "uuid-string",
  "name": "我的会话",
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-01T00:00:00",
  "documents": [],
  "messages": []
}
```

### 上传文档

```http
POST /sessions/{session_id}/documents
Content-Type: multipart/form-data

file: <file-content>
```

响应：
```json
{
  "id": "doc-uuid",
  "filename": "document.pdf",
  "chunk_count": 10
}
```

## 聊天接口

### 普通聊天

```http
POST /chat
Content-Type: application/json

{
  "message": "文档讲了什么？",
  "session_id": "session-uuid",
  "stream": false
}
```

响应：
```json
{
  "message": "根据文档内容...",
  "sources": [
    {
      "content": "文档段落...",
      "score": 0.5,
      "metadata": {}
    }
  ],
  "is_relevant": true
}
```

### 流式聊天

```http
POST /chat/stream
Content-Type: application/json

{
  "message": "文档讲了什么？",
  "session_id": "session-uuid",
  "stream": true
}
```

响应（SSE）：
```
data: {"delta": "根据", "finished": false}
data: {"delta": "文档", "finished": false}
data: {"delta": "内容", "finished": false}
data: {"delta": "", "finished": true, "sources": [...]}
```

## Agent 接口

### Agent 聊天

```http
POST /agent/chat
Content-Type: application/json

{
  "message": "计算 1 + 2，并告诉我现在几点",
  "session_id": "session-uuid"
}
```

响应：
```json
{
  "final_answer": "1 + 2 = 3，当前时间是...",
  "thoughts": [
    {
      "thought": "用户需要计算和获取时间",
      "action": {
        "tool": "calculate",
        "input": "1 + 2",
        "output": "计算结果: 3"
      }
    }
  ],
  "sources": []
}
```

## WebSocket 接口（计划中）

未来版本将支持 WebSocket 实时通信。
