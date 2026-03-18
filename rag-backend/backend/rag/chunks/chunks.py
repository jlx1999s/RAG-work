import re
from typing import List, Optional, Any, Union, Dict, Tuple

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from .models import ChunkStrategy, ChunkConfig, ChunkResult, DocumentContent


class TextChunker:
    """文本分块处理器
    
    支持多种分块策略：
    1. 字符级分块 - 按指定字符数分割
    2. 语义分块 - 基于语义相似度分割
    3. 递归分块 - 按分隔符优先级递归分割
    4. Markdown标题分块 - 按Markdown标题结构分割
    """
    
    def __init__(self, embeddings_model: Optional[Embeddings] = None):
        """初始化分块器
        
        Args:
            embeddings_model: 嵌入模型，语义分块时必需
        """
        self.embeddings_model = embeddings_model
    
    def chunk_document(self, document: DocumentContent, config: ChunkConfig) -> ChunkResult:
        """对文档进行分块处理
        
        Args:
            document: 文档内容对象
            config: 分块配置
            
        Returns:
            ChunkResult: 分块结果对象
        """
        if not document.content or not document.content.strip():
            return ChunkResult(
                chunks=[],
                strategy=config.strategy,
                total_chunks=0,
                document_name=document.document_name
            )
        
        try:
            if config.strategy == ChunkStrategy.CHARACTER:
                return self._character_chunk(document.content, config, document.document_name)
            elif config.strategy == ChunkStrategy.SEMANTIC:
                return self._semantic_chunk(document.content, config, document.document_name)
            elif config.strategy == ChunkStrategy.RECURSIVE:
                return self._recursive_chunk(document.content, config, document.document_name)
            elif config.strategy == ChunkStrategy.MARKDOWN_HEADER:
                return self._markdown_header_chunk(document.content, config, document.document_name)
            elif config.strategy == ChunkStrategy.MEDICAL_HYBRID:
                return self._medical_hybrid_chunk(document.content, config, document.document_name)
            else:
                return ChunkResult(
                    chunks=[],
                    strategy=config.strategy,
                    total_chunks=0,
                    document_name=document.document_name
                )
                
        except Exception as e:
            return ChunkResult(
                chunks=[],
                strategy=config.strategy,
                total_chunks=0,
                document_name=document.document_name
            )
    
    def _character_chunk(self, text: str, config: ChunkConfig, document_name: str) -> ChunkResult:
        """字符级分块"""
        text_splitter = CharacterTextSplitter(
            separator=config.separator,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            is_separator_regex=config.is_separator_regex,
        )
        
        chunks = text_splitter.create_documents([text])
        
        chunks = [Document(page_content=chunk.page_content, metadata=dict(chunk.metadata or {})) for chunk in chunks]
        
        return ChunkResult(
            chunks=chunks,
            strategy=config.strategy,
            total_chunks=len(chunks),
            document_name=document_name
        )
    
    def _semantic_chunk(self, text: str, config: ChunkConfig, document_name: str) -> ChunkResult:
        """语义分块"""
        if self.embeddings_model is None:
            return ChunkResult(
                chunks=[],
                strategy=config.strategy,
                total_chunks=0,
                document_name=document_name
            )
        
        semantic_chunker = SemanticChunker(
            embeddings=self.embeddings_model,
            breakpoint_threshold_type=config.breakpoint_threshold_type,
            breakpoint_threshold_amount=config.breakpoint_threshold_amount,
            sentence_split_regex=config.sentence_split_regex,
        )
        
        # 如果设置了最小分块大小
        if config.min_chunk_size:
            semantic_chunker.min_chunk_size = config.min_chunk_size
        
        chunks = semantic_chunker.create_documents([text])
        
        chunks = [Document(page_content=chunk.page_content, metadata=dict(chunk.metadata or {})) for chunk in chunks]
        
        return ChunkResult(
            chunks=chunks,
            strategy=config.strategy,
            total_chunks=len(chunks),
            document_name=document_name
        )
    
    def _recursive_chunk(self, text: str, config: ChunkConfig, document_name: str) -> ChunkResult:
        """递归分块"""
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=config.separators
        )
        
        chunks = recursive_splitter.create_documents([text])
        
        chunks = [Document(page_content=chunk.page_content, metadata=dict(chunk.metadata or {})) for chunk in chunks]
        
        return ChunkResult(
            chunks=chunks,
            strategy=config.strategy,
            total_chunks=len(chunks),
            document_name=document_name
        )
    
    def _markdown_header_chunk(self, text: str, config: ChunkConfig, document_name: str) -> ChunkResult:
        """Markdown标题分块"""
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=config.headers_to_split_on
        )
        
        chunks = markdown_splitter.split_text(text)
        
        # 转换为Document对象并保留结构metadata
        if isinstance(chunks, list) and chunks:
            if isinstance(chunks[0], Document):
                documents = [Document(page_content=doc.page_content, metadata=dict(doc.metadata or {})) for doc in chunks]
            else:
                documents = [Document(page_content=chunk, metadata={}) for chunk in chunks]
        else:
            documents = []
        
        return ChunkResult(
            chunks=documents,
            strategy=config.strategy,
            total_chunks=len(documents),
            document_name=document_name
        )

    def _medical_hybrid_chunk(self, text: str, config: ChunkConfig, document_name: str) -> ChunkResult:
        sections = self._split_medical_sections(text, config)
        documents: List[Document] = []
        for section_title, section_text in sections:
            section_chunks = self._split_medical_section_to_chunks(section_text, config)
            for chunk_text in section_chunks:
                cleaned_chunk = chunk_text.strip()
                if not cleaned_chunk:
                    continue
                metadata = self._build_medical_metadata(section_title, cleaned_chunk)
                documents.append(Document(page_content=cleaned_chunk, metadata=metadata))
        return ChunkResult(
            chunks=documents,
            strategy=config.strategy,
            total_chunks=len(documents),
            document_name=document_name
        )

    def _split_medical_sections(self, text: str, config: ChunkConfig) -> List[Tuple[str, str]]:
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=config.headers_to_split_on)
        split_docs = markdown_splitter.split_text(text)
        sections: List[Tuple[str, str]] = []
        if split_docs:
            for doc in split_docs:
                if not isinstance(doc, Document):
                    continue
                section_title = (
                    doc.metadata.get("Header_6")
                    or doc.metadata.get("Header_5")
                    or doc.metadata.get("Header_4")
                    or doc.metadata.get("Header_3")
                    or doc.metadata.get("Header_2")
                    or doc.metadata.get("Header_1")
                    or "未标注章节"
                )
                content = (doc.page_content or "").strip()
                if content:
                    sections.append((section_title, content))
        if not sections:
            sections = [("未标注章节", text)]
        merged_sections: List[Tuple[str, str]] = []
        for title, content in sections:
            if not merged_sections:
                merged_sections.append((title, content))
                continue
            if len(content) < config.medical_min_section_length:
                prev_title, prev_content = merged_sections[-1]
                merged_sections[-1] = (prev_title, f"{prev_content}\n{content}")
            else:
                merged_sections.append((title, content))
        return merged_sections

    def _split_medical_section_to_chunks(self, section_text: str, config: ChunkConfig) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.medical_chunk_size,
            chunk_overlap=config.medical_chunk_overlap,
            length_function=len,
            separators=config.separators
        )
        docs = splitter.create_documents([section_text])
        return [doc.page_content for doc in docs if (doc.page_content or "").strip()]

    def _build_medical_metadata(self, section_title: str, chunk_text: str) -> Dict[str, Any]:
        title = (section_title or "未标注章节").strip()
        content = chunk_text or ""
        merged = f"{title}\n{content}".lower()
        keyword_roles = [
            ("contraindication", ["禁忌", "禁用", "慎用", "不适用", "妊娠", "哺乳"]),
            ("dosage", ["用法", "用量", "剂量", "给药", "滴定", "频次", "疗程"]),
            ("indication", ["适应症", "适用人群", "诊断标准", "纳入标准"]),
            ("adverse_reaction", ["不良反应", "副作用", "风险", "并发症"]),
            ("recommendation", ["推荐", "证据", "共识", "指南", "级别", "等级"])
        ]
        chunk_role = "general_medical"
        for role, keywords in keyword_roles:
            if any(keyword in merged for keyword in keywords):
                chunk_role = role
                break
        evidence_match = re.search(r"(?:证据等级|推荐等级|grade|class)\s*[:：]?\s*([A-D][+-]?|I{1,3}[ab]?)", merged, re.IGNORECASE)
        evidence_level = evidence_match.group(1).upper() if evidence_match else None
        is_key_clause = chunk_role in {"contraindication", "dosage", "recommendation"}
        return {
            "section_title": title,
            "chunk_role": chunk_role,
            "evidence_level": evidence_level,
            "is_key_clause": is_key_clause
        }
    
    def chunk_with_strategy(self, text: str, strategy: Union[str, ChunkStrategy], 
                           document_name: str = "", **kwargs) -> ChunkResult:
        """便捷的分块方法
        
        Args:
            text: 待分块的文本
            strategy: 分块策略
            **kwargs: 分块参数
            
        Returns:
            ChunkResult: 分块结果
        """
        if isinstance(strategy, str):
            try:
                strategy = ChunkStrategy(strategy)
            except ValueError:
                return ChunkResult(
                    chunks=[],
                    strategy=ChunkStrategy.CHARACTER,
                    total_chunks=0,
                    document_name=document_name
                )
        
        config = ChunkConfig(strategy=strategy, **kwargs)
        document = DocumentContent(content=text, document_name=document_name)
        return self.chunk_document(document, config)


# 使用示例
if __name__ == "__main__":

    
    # 示例文本
    sample_text = """
    # 人工智能简介
    
    人工智能（Artificial Intelligence，AI）是计算机科学的一个分支。它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    
    ## 发展历史
    
    人工智能的研究历史已有数十年。从1956年的达特茅斯会议开始，人工智能正式成为一门学科。
    
    ### 早期发展
    
    早期的人工智能研究主要集中在逻辑推理和专家系统上。研究者们试图通过编写规则来模拟人类的推理过程。
    
    ### 现代发展
    
    随着机器学习和深度学习技术的发展，人工智能取得了突破性进展。特别是在图像识别、自然语言处理和游戏AI等领域。
    
    ## 应用领域
    
    人工智能目前在多个领域都有广泛应用：医疗诊断、自动驾驶、智能推荐系统等。这些应用正在改变我们的生活方式。
    """
    
    # 创建分块器（不使用嵌入模型进行基本测试）
    chunker = TextChunker()
    
    print("=== 字符级分块测试 ===")
    char_config = ChunkConfig(
        strategy=ChunkStrategy.CHARACTER,
        chunk_size=50,
        chunk_overlap=10,
        separator=""
    )
    char_result = chunker.chunk_document(DocumentContent(content=sample_text, document_name="test.txt"), char_config)
    print(f"总块数: {char_result.total_chunks}")
    print(char_result)
    # if char_result.chunks:
    #     print(f"第一块内容: {char_result.chunks[0].page_content[:100]}...")
    
    # print("\n=== 递归分块测试 ===")
    # recursive_config = ChunkConfig(
    #     strategy=ChunkStrategy.RECURSIVE,
    #     chunk_size=100,
    #     chunk_overlap=20
    # )
    # recursive_result = chunker.chunk_text(sample_text, recursive_config)
    # print(f"成功: {recursive_result.success}")
    # print(f"总块数: {recursive_result.total_chunks}")
    # if recursive_result.chunks:
    #     print(f"第一块内容: {recursive_result.chunks[0].page_content[:100]}...")
    
    # print("\n=== Markdown标题分块测试 ===")
    # md_config = ChunkConfig(
    #     strategy=ChunkStrategy.MARKDOWN_HEADER
    # )
    # md_result = chunker.chunk_text(sample_text, md_config)
    # print(f"成功: {md_result.success}")
    # print(f"总块数: {md_result.total_chunks}")
    # if md_result.chunks:
    #     print(f"第一块内容: {md_result.chunks[0].page_content[:100]}...")
    
    # print("\n=== 便捷方法测试 ===")
    # quick_result = chunker.chunk_with_strategy(
    #     sample_text, 
    #     "recursive",
    #     chunk_size=150,
    #     chunk_overlap=30
    # )
    # print(f"成功: {quick_result.success}")
    # print(f"总块数: {quick_result.total_chunks}")
    
    # # 测试语义分块（需要嵌入模型）
    # print("\n=== 语义分块测试（需要嵌入模型）===")
    # try:
    #     # 注意：这里需要先注册嵌入模型提供商
    #     # register_embeddings_provider("openai", "openai")  
    #     # embeddings_model = load_embeddings("openai:text-embedding-ada-002")
        
    #     # 这里使用None模型测试错误处理
    #     semantic_config = ChunkConfig(
    #         strategy=ChunkStrategy.SEMANTIC,
    #         breakpoint_threshold_amount=90
    #     )
    #     semantic_result = chunker.chunk_text(sample_text, semantic_config)
    #     print(f"成功: {semantic_result.success}")
    #     print(f"错误信息: {semantic_result.error_message}")
        
    # except Exception as e:
    #     print(f"语义分块测试失败: {e}")
