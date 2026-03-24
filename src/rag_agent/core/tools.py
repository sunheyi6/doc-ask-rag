"""
Agent 工具模块 - 定义可调用工具
"""
from typing import Any, Dict, Callable
import json
import re
from datetime import datetime
from .vectorstore import VectorStore


class Tool:
    """工具基类"""
    
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
    
    def invoke(self, input_str: str) -> str:
        """调用工具"""
        try:
            return self.func(input_str)
        except Exception as e:
            return f"工具调用失败: {str(e)}"


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """注册工具"""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Tool:
        """获取工具"""
        return self._tools.get(name)
    
    def list_tools(self) -> str:
        """列出所有工具"""
        return "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self._tools.items()
        ])
    
    def get_tool_schemas(self) -> list:
        """获取工具模式（用于 OpenAI function calling）"""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "工具输入参数"
                            }
                        },
                        "required": ["input"]
                    }
                }
            }
            for name, tool in self._tools.items()
        ]


# ========== 工具函数定义 ==========

def calculate(expression: str) -> str:
    """
    计算数学表达式
    
    支持: +, -, *, /, **, %, //
    示例: "1 + 2", "3 * 4", "10 / 2"
    """
    try:
        # 安全检查：只允许数学运算符和数字
        allowed_chars = set('0123456789+-*/.() **%// ')
        if not all(c in allowed_chars for c in expression):
            return "错误: 表达式包含非法字符"
        
        # 计算结果
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


def get_current_time(_: str = "") -> str:
    """获取当前日期和时间"""
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')}"


def create_knowledge_base_tool(vectorstore: VectorStore):
    """创建知识库检索工具"""
    
    def search_documents(query: str) -> str:
        """
        在知识库中搜索相关文档
        
        当用户询问文档内容时使用此工具
        """
        results = vectorstore.similarity_search(query, k=3)
        
        if not results:
            return "未找到相关文档"
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append(
                f"[相关度: {score:.3f}]\n{doc.page_content[:500]}..."
            )
        
        return "\n\n---\n\n".join(formatted_results)
    
    return search_documents


def web_search(query: str) -> str:
    """
    网络搜索（模拟）
    
    当知识库中没有相关信息时，可以搜索互联网获取答案
    注意：此为模拟实现，实际使用时需要接入真实搜索 API
    """
    # 这里应该接入真实的搜索 API，如 Bing、Google 等
    return f"[模拟搜索] 搜索结果: 关于 '{query}' 的更多信息建议查阅相关网站"


# ========== 创建默认工具注册表 ==========

def create_default_tools(vectorstore: VectorStore = None) -> ToolRegistry:
    """创建默认工具集"""
    registry = ToolRegistry()
    
    # 注册计算工具
    registry.register(Tool(
        name="calculate",
        description="计算数学表达式，支持 +, -, *, /, ** 等运算符",
        func=calculate
    ))
    
    # 注册时间工具
    registry.register(Tool(
        name="get_current_time",
        description="获取当前日期和时间",
        func=get_current_time
    ))
    
    # 注册知识库搜索工具（如果有向量库）
    if vectorstore is not None:
        registry.register(Tool(
            name="search_documents",
            description="在已上传的文档中搜索相关信息",
            func=create_knowledge_base_tool(vectorstore)
        ))
    
    # 注册网络搜索工具
    registry.register(Tool(
        name="web_search",
        description="当知识库中没有答案时，搜索互联网获取信息",
        func=web_search
    ))
    
    return registry
