"""
RAG Chain 模块 - 整合检索和生成的完整流程
"""
from typing import Iterator, Optional, Dict, Any, List
from dataclasses import dataclass
from langchain_core.documents import Document
from .vectorstore import VectorStore
from .llm import LLM, RAGPromptBuilder
from ..config import config
import re


@dataclass
class RAGResponse:
    """RAG 响应结果"""
    answer: str
    sources: List[Dict[str, Any]]  # 引用来源
    is_relevant: bool  # 是否与文档相关
    is_identity_question: bool = False  # 是否是身份问候类问题
    

class RAGChain:
    """RAG 问答链"""
    
    def __init__(self, vectorstore: VectorStore = None):
        self.vectorstore = vectorstore or VectorStore()
        self.llm = LLM()
    
    # 预设回答
    IDENTITY_ANSWER = "我是小A，您的个人知识库助手，我可以根据您上传的文档为您解答问题。"
    
    CAPABILITY_ANSWER = """我是小A，您的个人知识库助手。我可以帮您：

📄 **文档问答**
- 上传 PDF、TXT、DOCX 格式的文档
- 针对文档内容回答您的问题
- 显示回答的引用来源

💡 **智能功能**
- 理解文档语义，不只是关键词匹配
- 自动判断问题是否与文档相关
- 支持连续多轮对话

📝 **支持的操作**
- 在左侧边栏上传文档
- 直接输入问题提问
- 查看"查看引用来源"了解回答依据

有什么关于您上传文档的问题，随时问我！"""
    
    LIMITATION_ANSWER = """作为小A，我有一些限制：

🚫 **我不能做的**
- 回答与上传文档无关的问题
- 访问互联网获取实时信息（除非使用搜索工具）
- 执行复杂的数学计算（但我可以调用计算工具）
- 记住跨会话的信息
- 处理图像、音频、视频内容（暂不支持）

✅ **我的设计原则**
- 只基于您上传的文档回答
- 不确定时会告知您
- 不会编造信息

如果您需要我处理文档之外的内容，建议先上传相关资料！"""

    @staticmethod
    def _normalize_text(text: str) -> str:
        """标准化文本，便于做稳健匹配"""
        return re.sub(r"\s+", "", (text or "").strip().lower())

    @staticmethod
    def _normalize_question_for_match(question: str) -> str:
        """标准化问句：去空白、统一标点、去末尾语气助词，提升口语问法命中率"""
        q = (question or "").strip().lower()
        q = re.sub(r"\s+", "", q)
        q = q.replace("？", "?").replace("！", "!").replace("。", ".")
        q = re.sub(r"[?!.]+$", "", q)
        q = re.sub(r"[呀啊呢嘛吧哈哦噢]+$", "", q)
        q = re.sub(r"[?!.]+$", "", q)
        return q

    def _is_meta_answer_text(self, answer: str) -> bool:
        """兜底判断回答文本是否属于助手身份/能力/限制类回答"""
        answer_norm = self._normalize_text(answer)
        identity_norm = self._normalize_text(self.IDENTITY_ANSWER)
        capability_norm = self._normalize_text(self.CAPABILITY_ANSWER)
        limitation_norm = self._normalize_text(self.LIMITATION_ANSWER)

        return (
            answer_norm == identity_norm
            or answer_norm == capability_norm
            or answer_norm == limitation_norm
            or answer_norm.startswith(self._normalize_text("我是小A，您的个人知识库助手"))
        )
    
    def _is_math_question(self, question: str) -> bool:
        """检测是否为简单数学问题"""
        # 检测数学表达式如 "1+1", "2*3" 等
        pattern = r'^\s*\d+\s*[+\-*/]\s*\d+\s*$'
        return bool(re.match(pattern, question.strip()))
    
    def _is_meta_question(self, question: str) -> tuple[bool, str]:
        """
        检测是否为关于助手自身的元问题（身份、能力等）
        
        Returns:
            (是否是元问题, 回答类型: 'identity' 或 'capability')
        """
        question_clean = self._normalize_question_for_match(question)
        
        # 身份/问候类
        identity_patterns = [
            r'^你是谁\??$',
            r'^你叫什么名字\??$',
            r'^你是什么\??$',
            r'^你好,?你是谁\??$',
            r'^自我介绍[一下]?[吧]?\??$',
            r'^hi,?你是谁\??$',
            r'^hello,?你是谁\??$',
            r'^你是[谁小]a\??$',
        ]
        
        # 能力/功能类（正向）
        capability_patterns = [
            r'^你能做什么\??$',
            r'^你会什么\??$',
            r'^你有什么功能\??$',
            r'^你可以做什么\??$',
            r'^你怎么用\??$',
            r'^你怎么使用\??$',
            r'^你有什么用\??$',
            r'^你是干什么的\??$',
            r'^你的作用是什么\??$',
            r'^你能帮我做什么\??$',
            r'^你能[做干]啥\??$',
            r'^你有啥功能\??$',
            r'^你[做干]啥[用的]\??$',
        ]
        
        # 能力/限制类（反向）
        limitation_patterns = [
            r'^你不能做什么\??$',
            r'^你不会什么\??$',
            r'^你有什么限制\??$',
            r'^你有什么局限\??$',
            r'^你不能[做干]啥\??$',
            r'^你[做干]不了什么\??$',
            r'^你的限制是什么\??$',
            r'^你不能帮我做什么\??$',
        ]
        
        for pattern in identity_patterns:
            if re.match(pattern, question_clean):
                return True, "identity"
        
        for pattern in capability_patterns:
            if re.match(pattern, question_clean):
                return True, "capability"
        
        for pattern in limitation_patterns:
            if re.match(pattern, question_clean):
                return True, "limitation"
        
        return False, ""

    def _is_ambiguous_identity_statement(self, question: str) -> bool:
        """
        检测易误判的身份陈述句（如“你是十一”）。
        这类输入经常被模型误解为“你是谁”，因此优先按无关问题处理。
        """
        question_clean = self._normalize_question_for_match(question)
        if not question_clean.startswith("你是"):
            return False

        # 明确的元问题不在此分支处理
        is_meta, _ = self._is_meta_question(question_clean)
        if is_meta:
            return False

        # 限制在简短陈述句，避免误伤正常的文档问题
        return bool(re.match(r"^你是[\u4e00-\u9fff0-9a-z]{1,8}\??$", question_clean))
    
    def _is_common_sense_question(self, question: str) -> bool:
        """检测是否为常识性问题（时间、天气、地理位置等）"""
        question_clean = self._normalize_question_for_match(question)
        
        # 时间相关
        time_patterns = [
            r'^(现在|目前|当前)?(几点|几|什么时间|什么时候|时间)[了]?\??$',
            r'^(今天|明天|昨天|现在)是?(几号|日期|星期几)[了]?\??$',
            r'^(现在|当前)?((什么|哪个)年|年份)[了]?\??$',
        ]
        
        # 天气相关
        weather_patterns = [
            r'^(今天|明天|现在)(天气|气温|温度)(怎么样|如何|多少)?\??$',
            r'^(今天|明天)(下雨|下雪|晴天|阴天)吗\??$',
        ]
        
        # 地理位置相关
        location_patterns = [
            r'^((我在|这是|这里是))?(哪里|什么地方|哪个城市|哪个国家)\??$',
        ]
        
        all_patterns = time_patterns + weather_patterns + location_patterns
        
        for pattern in all_patterns:
            if re.search(pattern, question_clean):
                return True
        return False
    
    def _check_relevance(self, question: str, target_filename: Optional[str] = None) -> tuple[bool, str]:
        """
        检查问题是否与文档相关
        
        Returns:
            (is_relevant, context)
        """
        # 常识性问题直接判定为不相关（不需要查文档）
        if self._is_common_sense_question(question):
            return False, ""
        
        # 简单数学问题直接判定为不相关
        if self._is_math_question(question) and len(question.strip()) < 20:
            return False, ""

        # 避免“你是xxx”被误判成身份问候并触发自我介绍
        if self._is_ambiguous_identity_statement(question):
            return False, ""
        
        metadata_filter = {"filename": target_filename} if target_filename else None

        # 获取相关上下文
        context = self.vectorstore.get_relevant_context(
            question,
            top_k=3,
            score_threshold=config.SIMILARITY_THRESHOLD,
            metadata_filter=metadata_filter
        )
        
        # 如果获取不到上下文，判定为不相关
        if not context:
            return False, ""
        
        return True, context
    
    def invoke(self, question: str, chat_history: str = "", target_filename: Optional[str] = None) -> RAGResponse:
        """
        执行 RAG 问答（同步）
        
        Args:
            question: 用户问题
            chat_history: 历史对话
            
        Returns:
            RAGResponse 包含答案和引用
        """
        # 检测元问题（关于助手自身的问题），直接返回预设回答，不检索文档
        is_meta, meta_type = self._is_meta_question(question)
        if is_meta:
            if meta_type == "identity":
                answer = self.IDENTITY_ANSWER
            elif meta_type == "limitation":
                answer = self.LIMITATION_ANSWER
            else:
                answer = self.CAPABILITY_ANSWER
            return RAGResponse(
                answer=answer,
                sources=[],  # 元问题不显示引用
                is_relevant=True,
                is_identity_question=True
            )
        
        # 检查相关性
        is_relevant, context = self._check_relevance(question, target_filename=target_filename)
        
        if not is_relevant:
            return RAGResponse(
                answer="您提问的问题与文档无关，请提问与文档相关问题",
                sources=[],
                is_relevant=False,
                is_identity_question=True  # 标记为 True 以避免显示引用来源
            )
        
        # 构建提示词
        prompt = RAGPromptBuilder.build(context, question, chat_history)
        
        # 调用 LLM
        answer = self.llm.invoke(prompt)

        # 兜底：若模型输出的是助手身份/能力/限制类回答，不展示引用来源
        if self._is_meta_answer_text(answer):
            return RAGResponse(
                answer=answer,
                sources=[],
                is_relevant=True,
                is_identity_question=True
            )
        
        # 获取来源信息
        sources = self._get_sources(question, target_filename=target_filename)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            is_relevant=True
        )
    
    def stream(self, question: str, chat_history: str = "", target_filename: Optional[str] = None) -> Iterator[str]:
        """
        流式执行 RAG 问答
        
        Args:
            question: 用户问题
            chat_history: 历史对话
            
        Yields:
            生成的文本片段
        """
        # 检测元问题，直接返回预设回答
        is_meta, meta_type = self._is_meta_question(question)
        if is_meta:
            if meta_type == "identity":
                answer = self.IDENTITY_ANSWER
            elif meta_type == "limitation":
                answer = self.LIMITATION_ANSWER
            else:
                answer = self.CAPABILITY_ANSWER
            yield answer
            return
        
        # 检查相关性
        is_relevant, context = self._check_relevance(question, target_filename=target_filename)
        
        if not is_relevant:
            yield "您提问的问题与文档无关，请提问与文档相关问题"
            return
        
        # 构建提示词
        prompt = RAGPromptBuilder.build(context, question, chat_history)
        
        # 流式调用 LLM
        for chunk in self.llm.stream(prompt):
            yield chunk
    
    def _get_sources(self, question: str, k: int = 3, target_filename: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取引用来源"""
        metadata_filter = {"filename": target_filename} if target_filename else None
        docs_with_scores = self.vectorstore.similarity_search(
            question,
            k=k,
            metadata_filter=metadata_filter
        )
        
        sources = []
        for doc, score in docs_with_scores[:k]:
            if score <= config.SIMILARITY_THRESHOLD:
                sources.append({
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "score": round(score, 3),
                    "metadata": doc.metadata
                })
        
        return sources
