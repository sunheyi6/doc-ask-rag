"""
ReAct Agent 模块 - 推理与行动循环
"""
import json
import re
from typing import List, Dict, Any, Iterator, Optional
from dataclasses import dataclass, field
from .llm import LLM
from .tools import ToolRegistry, create_default_tools, Tool
from .vectorstore import VectorStore
from ..config import config


@dataclass
class AgentStep:
    """Agent 执行步骤"""
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None
    is_final: bool = False


@dataclass
class AgentResponse:
    """Agent 响应"""
    final_answer: str
    steps: List[AgentStep] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)


class ReActAgent:
    """
    ReAct Agent - 推理与行动循环
    
    实现思路:
    1. 观察用户输入
    2. 思考 (Thought) - 决定下一步行动
    3. 行动 (Action) - 调用工具
    4. 观察 (Observation) - 获取工具输出
    5. 重复直到得出最终答案
    """
    
    SYSTEM_PROMPT = """你是一个智能助手，可以使用工具帮助用户解决问题。

你可以使用以下工具:
{tools}

请按照以下格式思考和回答:

思考: 你需要思考如何解决问题
行动: 工具名称
输入: 工具的输入参数

工具会返回观察结果，然后你可以继续思考或给出最终答案:

观察: [工具返回的结果]

当你有最终答案时，请使用以下格式:

思考: 我已经找到答案
最终答案: 你的回答

注意:
- 如果问题与文档相关，优先使用 search_documents 工具
- 如果需要计算，使用 calculate 工具
- 如果需要知道当前时间，使用 get_current_time 工具
- 如果文档中没有相关信息，可以使用 web_search 工具
"""
    
    def __init__(self, llm: LLM = None, tools: ToolRegistry = None):
        self.llm = llm or LLM()
        self.tools = tools or create_default_tools()
        self.max_iterations = 5
    
    def _format_tools(self) -> str:
        """格式化工具描述"""
        return self.tools.list_tools()
    
    def _parse_response(self, text: str) -> Dict[str, str]:
        """解析 LLM 响应"""
        result = {
            "thought": "",
            "action": None,
            "action_input": None,
            "final_answer": None
        }
        
        # 提取思考
        thought_match = re.search(r'思考:\s*(.+?)(?=\n行动:|\n最终答案:|$)', text, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        # 提取行动
        action_match = re.search(r'行动:\s*(\w+)', text)
        if action_match:
            result["action"] = action_match.group(1).strip()
        
        # 提取输入
        input_match = re.search(r'输入:\s*(.+?)(?=\n观察:|\n思考:|$)', text, re.DOTALL)
        if input_match:
            result["action_input"] = input_match.group(1).strip()
        
        # 提取最终答案
        final_match = re.search(r'最终答案:\s*(.+)', text, re.DOTALL)
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
        
        return result
    
    def invoke(self, query: str, chat_history: str = "") -> AgentResponse:
        """
        执行 Agent 推理循环
        
        Args:
            query: 用户查询
            chat_history: 历史对话
            
        Returns:
            AgentResponse 包含最终答案和执行步骤
        """
        steps = []
        prompt = self._build_initial_prompt(query, chat_history)
        
        for i in range(self.max_iterations):
            # 调用 LLM
            response = self.llm.invoke(prompt)
            
            # 解析响应
            parsed = self._parse_response(response)
            
            # 如果有最终答案，返回
            if parsed["final_answer"]:
                step = AgentStep(
                    thought=parsed["thought"],
                    is_final=True
                )
                steps.append(step)
                
                return AgentResponse(
                    final_answer=parsed["final_answer"],
                    steps=steps
                )
            
            # 执行工具调用
            if parsed["action"] and parsed["action_input"]:
                tool = self.tools.get(parsed["action"])
                if tool:
                    observation = tool.invoke(parsed["action_input"])
                else:
                    observation = f"错误: 未知工具 '{parsed['action']}'"
                
                step = AgentStep(
                    thought=parsed["thought"],
                    action=parsed["action"],
                    action_input=parsed["action_input"],
                    observation=observation
                )
                steps.append(step)
                
                # 更新 prompt，添加观察结果
                prompt += f"\n{response}\n观察: {observation}\n\n"
            else:
                # 没有行动和最终答案，可能是格式问题
                step = AgentStep(
                    thought=parsed["thought"] or response,
                    is_final=True
                )
                steps.append(step)
                
                return AgentResponse(
                    final_answer=response,
                    steps=steps
                )
        
        # 达到最大迭代次数
        return AgentResponse(
            final_answer="抱歉，我尝试了多次但无法得出结论。",
            steps=steps
        )
    
    def _build_initial_prompt(self, query: str, chat_history: str) -> str:
        """构建初始提示词"""
        tools_desc = self._format_tools()
        
        prompt = self.SYSTEM_PROMPT.format(tools=tools_desc)
        
        if chat_history:
            prompt += f"\n\n历史对话:\n{chat_history}\n"
        
        prompt += f"\n用户问题: {query}\n\n"
        
        return prompt


class RAGAgent(ReActAgent):
    """
    RAG + Agent 组合
    
    先尝试使用 RAG 检索回答，如果不够再调用 Agent 工具
    """
    
    def __init__(self, vectorstore: VectorStore, llm: LLM = None):
        tools = create_default_tools(vectorstore)
        super().__init__(llm, tools)
        self.vectorstore = vectorstore
    
    def invoke(self, query: str, chat_history: str = "") -> AgentResponse:
        """
        RAG + Agent 组合调用
        
        策略:
        1. 先尝试 RAG 检索
        2. 如果检索到相关信息，直接回答
        3. 如果检索不到或需要计算/搜索，使用 Agent
        """
        # 首先尝试 RAG 检索
        context = self.vectorstore.get_relevant_context(
            query,
            top_k=3,
            score_threshold=config.SIMILARITY_THRESHOLD
        )
        
        if context:
            # 有相关知识，使用 RAG 回答
            prompt = f"""基于以下上下文回答用户问题：

上下文：
{context}

用户问题：{query}

请直接回答，如果上下文不足以回答问题，请说明。"""
            
            answer = self.llm.invoke(prompt)
            
            return AgentResponse(
                final_answer=answer,
                steps=[AgentStep(
                    thought="从知识库中检索到相关信息，直接回答",
                    is_final=True
                )]
            )
        
        # 没有相关知识，使用 Agent 工具
        return super().invoke(query, chat_history)
