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


class TestBM25Index:
    """测试手写 BM25 关键词索引（混合检索的关键词检索器）"""

    def _make_docs(self):
        from langchain_core.documents import Document
        return [
            Document(page_content="ETQ 系统的部署手册：需要 Java 17 环境，使用 Docker 部署", metadata={"filename": "deploy.txt"}),
            Document(page_content="公司团建活动安排：本周五下午在公园烧烤，欢迎报名参加", metadata={"filename": "party.txt"}),
            Document(page_content="Java 开发规范：方法命名使用驼峰，类名首字母大写", metadata={"filename": "java_style.txt"}),
        ]

    def test_tokenize_mixed(self):
        """测试中英文混合分词"""
        from rag_agent.core.vectorstore import _tokenize
        tokens = _tokenize("ETQ系统部署")
        assert "etq" in tokens       # 英文按词
        assert "系统" in tokens      # 中文 2-gram
        assert "部署" in tokens

    def test_search_keyword_hit(self):
        """测试关键词检索：能精确命中专有名词"""
        from rag_agent.core.vectorstore import BM25Index
        index = BM25Index(self._make_docs())
        results = index.search("ETQ 部署")
        assert len(results) > 0
        # 含 ETQ 的文档应排第一
        assert results[0][0].metadata["filename"] == "deploy.txt"

    def test_search_no_hit(self):
        """测试无关键词命中时返回空"""
        from rag_agent.core.vectorstore import BM25Index
        index = BM25Index(self._make_docs())
        results = index.search("量子力学弦理论")
        assert results == []

    def test_search_metadata_predicate(self):
        """测试按文件过滤检索"""
        from rag_agent.core.vectorstore import BM25Index
        index = BM25Index(self._make_docs())
        results = index.search(
            "Java",
            metadata_predicate=lambda doc: doc.metadata.get("filename") == "party.txt",
        )
        assert results == []  # party.txt 不含 Java


class TestRRFFusion:
    """测试 RRF 融合（混合检索的排序合并）"""

    def test_doc_in_both_lists_ranks_first(self):
        """两路检索都命中的文档应排在最前"""
        from langchain_core.documents import Document
        from rag_agent.core.vectorstore import _rrf_fuse
        a = Document(page_content="文档A：向量与BM25都命中", metadata={"filename": "a.txt"})
        b = Document(page_content="文档B：只有向量命中", metadata={"filename": "b.txt"})
        c = Document(page_content="文档C：只有BM25命中", metadata={"filename": "c.txt"})
        fused = _rrf_fuse([[(a, 0.1), (b, 0.2)], [(a, 5.0), (c, 3.0)]])
        assert fused[0][0] == a  # 双路命中 → 融合分最高
        assert [doc.page_content for doc, _ in fused] == [a.page_content, b.page_content, c.page_content]

    def test_rank_normalization(self):
        """验证融合分数 = Σ 1/(k + rank + 1)"""
        from langchain_core.documents import Document
        from rag_agent.core.vectorstore import _rrf_fuse
        a = Document(page_content="唯一文档", metadata={"filename": "a.txt"})
        fused = _rrf_fuse([[(a, 0.5)]], constant_k=60)
        assert abs(fused[0][1] - 1.0 / 61.0) < 1e-9


class TestHybridRetrieval:
    """测试 VectorStore 混合检索集成"""

    def test_hybrid_search_without_vectorstore(self):
        """未初始化向量库时，混合检索应优雅退化为仅 BM25"""
        from langchain_core.documents import Document
        from rag_agent.core.vectorstore import VectorStore
        vs = VectorStore(persist_directory="./tmp_test_no_vector")
        vs._bm25 = None
        # 无向量库也无 BM25 → 返回空列表而不是报错
        assert vs.hybrid_search("测试", k=3, score_threshold=1.8) == []

    def test_hybrid_search_bm25_only(self):
        """无向量库但有 BM25 索引时，融合结果来自 BM25"""
        from langchain_core.documents import Document
        from rag_agent.core.vectorstore import VectorStore, BM25Index
        docs = [Document(page_content="Apache License 2.0 是开源许可证", metadata={"filename": "lic.txt"})]
        vs = VectorStore(persist_directory="./tmp_test_bm25_only")
        vs._bm25 = BM25Index(docs)
        results = vs.hybrid_search("Apache License", k=3, score_threshold=1.8)
        assert len(results) == 1
        assert results[0][0].page_content == docs[0].page_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
