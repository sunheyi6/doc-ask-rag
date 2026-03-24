"""
LLM 模块 - 封装大模型调用，支持流式输出
"""
from typing import Iterator, Optional, Dict, Any
import dashscope
from ..config import config


class LLM:
    """大语言模型封装"""
    
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or config.LLM_MODEL
        self.api_key = api_key or config.DASHSCOPE_API_KEY
        dashscope.api_key = self.api_key
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """
        同步调用 LLM
        
        Args:
            prompt: 提示词
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        messages = [{"role": "user", "content": prompt}]
        
        response = dashscope.Generation.call(
            model=self.model,
            messages=messages,
            result_format="message",
            **kwargs
        )
        
        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            raise RuntimeError(f"LLM 调用失败: {response.message}")
    
    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        流式调用 LLM
        
        Args:
            prompt: 提示词
            **kwargs: 其他参数
            
        Yields:
            生成的文本片段（增量）
        """
        messages = [{"role": "user", "content": prompt}]
        
        responses = dashscope.Generation.call(
            model=self.model,
            messages=messages,
            result_format="message",
            stream=True,
            **kwargs
        )
        
        previous_text = ""
        for response in responses:
            if response.status_code == 200:
                current_text = response.output.choices[0].message.content
                if current_text:
                    # 计算增量（delta）
                    delta = current_text[len(previous_text):]
                    if delta:
                        yield delta
                    previous_text = current_text
            else:
                raise RuntimeError(f"LLM 流式调用失败: {response.message}")


class RAGPromptBuilder:
    """RAG 提示词构建器"""
    
    SYSTEM_PROMPT = """你是小A，用户的个人知识库助手。你的任务是根据上传的文档内容回答用户问题。

角色设定：
- 你的名字是小A
- 你是用户的专属知识库助手
- 当用户问"你是谁"时，请回答："我是小A，您的个人知识库助手，我可以根据您上传的文档为您解答问题。"

回答规则：
1. 如果上下文中包含相关信息，请基于文档内容直接回答
2. 只有当上下文完全没有相关信息时，才回答"您提问的问题与文档无关，请提问与文档相关问题"
3. 不要编造信息
4. 回答时请引用文档中的来源段落"""
    
    @classmethod
    def build(cls, context: str, question: str, chat_history: str = "") -> str:
        """构建 RAG 提示词"""
        history_part = f"\n\n历史对话：\n{chat_history}" if chat_history else ""
        
        return f"""{cls.SYSTEM_PROMPT}

文档上下文：
{context}{history_part}

用户问题：
{question}

请回答："""
