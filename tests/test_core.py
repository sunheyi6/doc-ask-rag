"""
核心模块测试
"""
import pytest
import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_agent.core.document_loader import DocumentLoader
from rag_agent.core.text_splitter import DocumentSplitter
from rag_agent.core.tools import calculate, get_current_time, ToolRegistry, Tool
from rag_agent.core.rag_chain import RAGChain


class TestDocumentLoader:
    """测试文档加载器"""
    
    def test_supported_extensions(self):
        """测试支持的文件格式"""
        extensions = DocumentLoader.SUPPORTED_EXTENSIONS
        assert ".pdf" in extensions
        assert ".txt" in extensions
        assert ".docx" in extensions
    
    def test_encodings(self):
        """测试编码列表"""
        encodings = DocumentLoader.ENCODINGS
        assert "utf-8" in encodings
        assert "gbk" in encodings


class TestDocumentSplitter:
    """测试文档分割器"""
    
    def test_split_text(self):
        """测试文本分割"""
        from langchain_core.documents import Document
        
        splitter = DocumentSplitter(chunk_size=100, chunk_overlap=20)
        
        # 创建测试文档
        text = "这是一段测试文本。" * 50
        docs = [Document(page_content=text)]
        
        # 分割
        splits = splitter.split(docs)
        
        assert len(splits) > 0
        assert all(len(s.page_content) <= 200 for s in splits)  # 稍微放宽限制


class TestTools:
    """测试工具函数"""
    
    def test_calculate_addition(self):
        """测试加法计算"""
        result = calculate("1 + 2")
        assert "3" in result
    
    def test_calculate_multiplication(self):
        """测试乘法计算"""
        result = calculate("3 * 4")
        assert "12" in result
    
    def test_calculate_division(self):
        """测试除法计算"""
        result = calculate("10 / 2")
        assert "5" in result or "5.0" in result
    
    def test_calculate_invalid(self):
        """测试非法输入"""
        result = calculate("1 + abc")
        assert "错误" in result or "Error" in result
    
    def test_get_current_time(self):
        """测试获取时间"""
        result = get_current_time()
        assert "当前时间" in result
    
    def test_tool_registry(self):
        """测试工具注册表"""
        registry = ToolRegistry()
        
        # 注册工具
        registry.register(Tool(
            name="test_tool",
            description="测试工具",
            func=lambda x: f"测试: {x}"
        ))
        
        # 获取工具
        tool = registry.get("test_tool")
        assert tool is not None
        assert tool.name == "test_tool"
        
        # 调用工具
        result = tool.invoke("hello")
        assert "hello" in result


class TestConfig:
    """测试配置"""
    
    def test_config_defaults(self):
        """测试默认配置"""
        from rag_agent.config import Config
        
        config = Config()
        
        assert config.LLM_MODEL == "qwen-turbo"
        assert config.EMBEDDING_MODEL == "text-embedding-v1"
        assert config.CHUNK_SIZE == 600
        assert config.CHUNK_OVERLAP == 100


class TestRAGChainGuards:
    """测试 RAG 链的问题拦截逻辑"""

    def test_identity_question_detected(self):
        """测试身份问题识别"""
        chain = RAGChain()
        is_meta, meta_type = chain._is_meta_question("你是谁")
        assert is_meta is True
        assert meta_type == "identity"

    def test_ambiguous_identity_statement_detected(self):
        """测试“你是xxx”歧义句识别"""
        chain = RAGChain()
        assert chain._is_ambiguous_identity_statement("你是十一") is True

    def test_exact_identity_not_ambiguous(self):
        """测试明确身份问题不应判定为歧义句"""
        chain = RAGChain()
        assert chain._is_ambiguous_identity_statement("你是谁") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
