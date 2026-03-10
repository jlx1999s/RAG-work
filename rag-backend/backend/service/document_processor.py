#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档处理器
专门处理用户上传的文档（与爬虫逻辑完全分离）
"""
import os
import re
from typing import Optional, Dict, Any, List, Tuple
import tempfile
from urllib.parse import urljoin
from langchain_core.documents import Document
from backend.config.log import get_logger
from backend.config.oss import OssClientFactory
from backend.rag.storage.milvus_storage import MilvusStorage
from backend.rag.storage.lightrag_storage import LightRAGStorage
from backend.rag.chunks.document_extraction import DocumentExtractor
from backend.rag.chunks.chunks import TextChunker
from backend.rag.chunks.models import ChunkConfig, ChunkStrategy, DocumentContent, ChunkResult

logger = get_logger(__name__)


class DocumentProcessor:
    """文档处理器
    
    负责处理用户上传的文档：
    1. 从 OSS 读取文档内容
    2. 分块处理
    3. 存储到向量数据库（Milvus）
    4. 构建知识图谱（LightRAG + Neo4j）
    """
    
    def __init__(self, collection_id: str, milvus_storage: MilvusStorage, lightrag_storage: LightRAGStorage):
        """初始化文档处理器
        
        Args:
            collection_id: 知识库 collection_id
            milvus_storage: Milvus 存储实例
            lightrag_storage: LightRAG 存储实例
        """
        self.collection_id = collection_id
        self.milvus_storage = milvus_storage
        self.lightrag_storage = lightrag_storage
        self.chunker = TextChunker()
        
    async def process_document(
        self,
        document_name: str,
        oss_bucket: str,
        oss_key: str,
        file_type: str = "md",
        source_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """处理文档的完整流程
        
        Args:
            document_name: 文档名称（用于 metadata）
            oss_bucket: OSS 存储桶名称
            oss_key: OSS 对象键
            file_type: 文件类型（md/txt/pdf 等）
            
        Returns:
            Dict: 处理结果
            
        Raises:
            Exception: 处理过程中的任何错误
        """
        logger.info(f"开始处理文档: {document_name}, OSS: {oss_bucket}/{oss_key}")
        
        try:
            # 1. 从 OSS 读取文档内容
            content = await self._read_from_oss(oss_bucket, oss_key, file_type, source_url)
            
            if not content or not content.strip():
                raise ValueError(f"文档内容为空: {document_name}")
            
            logger.info(f"成功从 OSS 读取文档，内容长度: {len(content)} 字符")
            
            # 2. 文档分块
            chunks_result = await self._chunk_document(content, document_name, file_type)
            chart_chunks_result = self._build_chart_chunks(content, document_name, source_url)
            all_chunk_results = [chunks_result]
            if chart_chunks_result.total_chunks > 0:
                all_chunk_results.append(chart_chunks_result)
            
            if not chunks_result or not chunks_result.chunks:
                raise ValueError(f"文档分块结果为空: {document_name}")
            
            text_chunk_count = len(chunks_result.chunks)
            chart_chunk_count = chart_chunks_result.total_chunks
            chunk_count = text_chunk_count + chart_chunk_count
            logger.info(f"文档分块完成，文本块: {text_chunk_count}, 图表块: {chart_chunk_count}")
            
            # 3. 存储到 Milvus
            await self._store_to_milvus(all_chunk_results, document_name)
            
            # 4. 构建知识图谱（LightRAG + Neo4j）
            await self._build_knowledge_graph(all_chunk_results, document_name)
            
            result = {
                "status": "success",
                "document_name": document_name,
                "chunk_count": chunk_count,
                "text_chunk_count": text_chunk_count,
                "chart_chunk_count": chart_chunk_count,
                "message": f"文档处理完成：{chunk_count} 个分块已入库"
            }
            
            logger.info(f"文档处理成功: {document_name}")
            return result
            
        except Exception as e:
            error_msg = f"处理文档失败: {document_name}, 错误: {str(e)}"
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
            raise Exception(error_msg)
    
    async def vectorize_document(
        self,
        document_name: str,
        oss_bucket: str,
        oss_key: str,
        file_type: str = "md",
        source_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """仅向量化文档（分块 + Milvus 存储）
        
        Args:
            document_name: 文档名称
            oss_bucket: OSS 存储桶
            oss_key: OSS 对象键
            file_type: 文件类型
            
        Returns:
            Dict: 处理结果
        """
        logger.info(f"开始向量化文档: {document_name}")
        
        try:
            # 1. 从 OSS 读取文档内容
            content = await self._read_from_oss(oss_bucket, oss_key, file_type, source_url)
            
            if not content or not content.strip():
                raise ValueError(f"文档内容为空: {document_name}")
            
            logger.info(f"成功从 OSS 读取文档，内容长度: {len(content)} 字符")
            
            # 2. 文档分块
            chunks_result = await self._chunk_document(content, document_name, file_type)
            chart_chunks_result = self._build_chart_chunks(content, document_name, source_url)
            all_chunk_results = [chunks_result]
            if chart_chunks_result.total_chunks > 0:
                all_chunk_results.append(chart_chunks_result)
            
            if not chunks_result or not chunks_result.chunks:
                raise ValueError(f"文档分块结果为空: {document_name}")
            
            text_chunk_count = len(chunks_result.chunks)
            chart_chunk_count = chart_chunks_result.total_chunks
            chunk_count = text_chunk_count + chart_chunk_count
            logger.info(f"文档分块完成，文本块: {text_chunk_count}, 图表块: {chart_chunk_count}")
            
            # 3. 存储到 Milvus
            await self._store_to_milvus(all_chunk_results, document_name)
            
            result = {
                "status": "success",
                "document_name": document_name,
                "chunk_count": chunk_count,
                "text_chunk_count": text_chunk_count,
                "chart_chunk_count": chart_chunk_count,
                "message": f"向量化完成：{chunk_count} 个分块已入库"
            }
            
            logger.info(f"文档向量化成功: {document_name}")
            return result
            
        except Exception as e:
            error_msg = f"向量化文档失败: {document_name}, 错误: {str(e)}"
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
            raise Exception(error_msg)
    
    async def graph_document(
        self,
        document_name: str,
        oss_bucket: str,
        oss_key: str,
        file_type: str = "md",
        source_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """仅图谱化文档（LightRAG + Neo4j）
        
        Args:
            document_name: 文档名称
            oss_bucket: OSS 存储桶
            oss_key: OSS 对象键
            file_type: 文件类型
            
        Returns:
            Dict: 处理结果
        """
        logger.info(f"开始图谱化文档: {document_name}")
        
        try:
            # 1. 从 OSS 读取文档内容
            content = await self._read_from_oss(oss_bucket, oss_key, file_type, source_url)
            
            if not content or not content.strip():
                raise ValueError(f"文档内容为空: {document_name}")
            
            logger.info(f"成功从 OSS 读取文档，内容长度: {len(content)} 字符")
            
            # 2. 文档分块
            chunks_result = await self._chunk_document(content, document_name, file_type)
            chart_chunks_result = self._build_chart_chunks(content, document_name, source_url)
            all_chunk_results = [chunks_result]
            if chart_chunks_result.total_chunks > 0:
                all_chunk_results.append(chart_chunks_result)
            
            if not chunks_result or not chunks_result.chunks:
                raise ValueError(f"文档分块结果为空: {document_name}")
            
            text_chunk_count = len(chunks_result.chunks)
            chart_chunk_count = chart_chunks_result.total_chunks
            chunk_count = text_chunk_count + chart_chunk_count
            logger.info(f"文档分块完成，文本块: {text_chunk_count}, 图表块: {chart_chunk_count}")
            
            # 3. 构建知识图谱
            await self._build_knowledge_graph(all_chunk_results, document_name)
            
            result = {
                "status": "success",
                "document_name": document_name,
                "chunk_count": chunk_count,
                "text_chunk_count": text_chunk_count,
                "chart_chunk_count": chart_chunk_count,
                "message": f"图谱化完成：{chunk_count} 个分块已入图"
            }
            
            logger.info(f"文档图谱化成功: {document_name}")
            return result
            
        except Exception as e:
            error_msg = f"图谱化文档失败: {document_name}, 错误: {str(e)}"
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
            raise Exception(error_msg)
    
    async def _read_from_oss(self, bucket: str, key: str, file_type: str, source_url: Optional[str]) -> str:
        """从 OSS 读取文档内容（使用 SDK，不走 HTTP）
        
        Args:
            bucket: OSS 存储桶
            key: 对象键
            
        Returns:
            str: 文档内容
        """
        try:
            import alibabacloud_oss_v2 as oss
            import urllib.parse
            
            client = OssClientFactory.get_client()
            
            # URL 解码 key（处理中文文件名）
            decoded_key = urllib.parse.unquote(key)
            
            # 使用 GetObject 直接读取内容
            request = oss.GetObjectRequest(bucket=bucket, key=decoded_key)
            result = client.get_object(request)
            
            raw_bytes = result.body.read()
            normalized_type = (file_type or "").lower()
            extractor = DocumentExtractor()
            detected_type = None
            if raw_bytes.startswith(b"%PDF"):
                detected_type = "pdf"
            elif raw_bytes.startswith(b"PK"):
                detected_type = "docx"
            elif raw_bytes[:4] == b"\xD0\xCF\x11\xE0":
                detected_type = "doc"
            if normalized_type in {"pdf", "doc", "docx"} or detected_type:
                effective_type = normalized_type if normalized_type in {"pdf", "doc", "docx"} else detected_type
                suffix = f".{effective_type}" if effective_type else ""
                with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as temp_file:
                    temp_file.write(raw_bytes)
                    temp_file.flush()
                    parsed = extractor.read_document(temp_file.name)
                    content = parsed.content
                if effective_type == "pdf" and (not content or not content.strip()):
                    mineru_url = os.getenv("MINERU_API_URL")
                    mineru_key = os.getenv("MINERU_API_KEY")
                    if mineru_url and mineru_key:
                        target_url = source_url
                        try:
                            from backend.config.oss import get_presigned_url_for_download
                            presign = get_presigned_url_for_download(bucket=bucket, key=decoded_key)
                            if presign and presign.get("url"):
                                target_url = presign.get("url")
                        except Exception as presign_error:
                            logger.warning(f"生成下载URL失败: {bucket}/{decoded_key}, 错误: {str(presign_error)}")
                        if target_url:
                            try:
                                parsed = extractor.read_document(target_url, pdf_extract_method="mineru")
                                content = parsed.content
                            except Exception as mineru_error:
                                logger.warning(f"MinerU 解析失败: {target_url}, 错误: {str(mineru_error)}")
            else:
                try:
                    content = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    content = raw_bytes.decode("utf-8", errors="ignore")
            
            logger.info(f"成功从 OSS 读取: {bucket}/{decoded_key}, 文件大小: {len(raw_bytes)} bytes, 内容长度: {len(content)}")
            return content
            
        except Exception as e:
            logger.error(f"从 OSS 读取失败: {bucket}/{key}, 错误: {str(e)}")
            raise Exception(f"OSS 读取失败: {str(e)}")
    
    async def _chunk_document(
        self,
        content: str,
        document_name: str,
        file_type: str
    ) -> Optional[Any]:
        """对文档内容进行分块
        
        Args:
            content: 文档内容
            document_name: 文档名称
            file_type: 文件类型
            
        Returns:
            ChunkResult: 分块结果
        """
        try:
            # 根据文件类型选择分块策略
            normalized_type = (file_type or "").lower()
            if normalized_type in ["md", "markdown"]:
                strategy = ChunkStrategy.MARKDOWN_HEADER
                chunk_config = ChunkConfig(strategy=strategy)
            else:
                strategy = ChunkStrategy.RECURSIVE
                chunk_config = ChunkConfig(strategy=strategy, chunk_size=1000, chunk_overlap=200)
            
            # 创建文档对象
            document = DocumentContent(
                content=content,
                document_name=document_name
            )
            
            # 执行分块
            result = self.chunker.chunk_document(document, chunk_config)
            for chunk in result.chunks:
                chunk.metadata = {
                    "chunk_type": "text",
                    "source": "vector"
                }
            
            logger.info(f"文档分块完成: {document_name}, 策略: {strategy.value}")
            return result
            
        except Exception as e:
            logger.error(f"文档分块失败: {document_name}, 错误: {str(e)}")
            raise
    
    def _extract_headings(self, content: str) -> List[Tuple[int, str]]:
        heading_matches = list(re.finditer(r"^#{1,6}\s+(.+)$", content, flags=re.MULTILINE))
        return [(match.start(), match.group(1).strip()) for match in heading_matches]

    def _find_closest_heading(self, headings: List[Tuple[int, str]], position: int) -> str:
        section_title = "未标注章节"
        for heading_pos, heading_title in headings:
            if heading_pos > position:
                break
            section_title = heading_title
        return section_title

    def _normalize_image_url(self, image_url: str, source_url: Optional[str]) -> str:
        if image_url.startswith(("http://", "https://")):
            return image_url
        if source_url:
            return urljoin(source_url, image_url)
        return image_url

    def _extract_page_number(self, context_text: str) -> Optional[int]:
        page_match = re.search(r"第\s*(\d+)\s*页|(?:P|p|Page)\s*[:：]?\s*(\d+)", context_text)
        if not page_match:
            return None
        page_value = page_match.group(1) or page_match.group(2)
        if not page_value:
            return None
        return int(page_value)

    def _sanitize_context(self, raw_text: str) -> str:
        cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", raw_text)
        cleaned = re.sub(r"<img[^>]+>", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _generate_chart_caption(self, alt_text: str, section_title: str, context_text: str) -> str:
        context = self._sanitize_context(context_text)
        context = context[:260] if context else ""
        alt = (alt_text or "").strip()
        if alt and context:
            return f"{section_title}；图示主题：{alt}；相关内容：{context}"
        if context:
            return f"{section_title}；相关内容：{context}"
        if alt:
            return f"{section_title}；图示主题：{alt}"
        return f"{section_title}；图表内容待补充"

    def _build_chart_chunks(self, content: str, document_name: str, source_url: Optional[str]) -> ChunkResult:
        headings = self._extract_headings(content)
        markdown_pattern = r"!\[(?P<alt>[^\]]*)\]\((?P<url>[^)\s]+)(?:\s+\"[^\"]*\")?\)"
        html_pattern = r"<img[^>]*src=[\"'](?P<url>[^\"']+)[\"'][^>]*>"
        all_matches = []
        all_matches.extend([("markdown", m) for m in re.finditer(markdown_pattern, content)])
        all_matches.extend([("html", m) for m in re.finditer(html_pattern, content, flags=re.IGNORECASE)])
        all_matches.sort(key=lambda item: item[1].start())
        chunks: List[Document] = []
        for idx, (_, match) in enumerate(all_matches, start=1):
            image_url = match.group("url").strip()
            alt_text = match.groupdict().get("alt", "").strip() if match.groupdict() else ""
            normalized_image_url = self._normalize_image_url(image_url, source_url)
            start = max(0, match.start() - 320)
            end = min(len(content), match.end() + 320)
            context_text = content[start:end]
            section_title = self._find_closest_heading(headings, match.start())
            page_number = self._extract_page_number(context_text)
            chart_id = f"{document_name}#chart-{idx}"
            caption = self._generate_chart_caption(alt_text, section_title, context_text)
            page_text = str(page_number) if page_number is not None else "未知"
            chunk_text = (
                f"图表ID: {chart_id}\n"
                f"所属章节: {section_title}\n"
                f"页码: {page_text}\n"
                f"图表摘要: {caption}\n"
                f"图表地址: {normalized_image_url}"
            )
            metadata = {
                "chunk_type": "chart",
                "source": "vector_chart",
                "chart_id": chart_id,
                "chart_image_url": normalized_image_url,
                "section_title": section_title,
                "page_number": page_number,
                "chart_caption": caption
            }
            chunks.append(Document(page_content=chunk_text, metadata=metadata))
        return ChunkResult(
            chunks=chunks,
            strategy=ChunkStrategy.RECURSIVE,
            total_chunks=len(chunks),
            document_name=document_name
        )

    async def _store_to_milvus(self, chunk_results: List[Any], document_name: str):
        """将分块存储到 Milvus
        
        Args:
            chunks_result: 分块结果
            document_name: 文档名称
        """
        try:
            result = self.milvus_storage.store_chunks_batch(chunk_results)
            
            logger.info(
                f"成功存储到 Milvus: {document_name}, "
                f"chunks: {result.get('total_chunks', 0)}, "
                f"collection: {self.collection_id}"
            )
            
        except Exception as e:
            logger.error(f"Milvus 存储失败: {document_name}, 错误: {str(e)}")
            raise Exception(f"向量存储失败: {str(e)}")
    
    async def _build_knowledge_graph(self, chunk_results: List[Any], document_name: str):
        """构建知识图谱（LightRAG + Neo4j）
        
        Args:
            chunks_result: 分块结果
            document_name: 文档名称
        """
        try:
            # 提取所有 chunk 的文本内容
            text_chunks = []
            for chunk_result in chunk_results:
                text_chunks.extend([chunk.page_content for chunk in chunk_result.chunks])
            
            # 初始化 LightRAG（如果尚未初始化）
            if self.lightrag_storage.rag is None:
                await self.lightrag_storage.initialize()
            
            # 批量插入文本到 LightRAG
            await self.lightrag_storage.insert_texts(text_chunks)
            
            logger.info(
                f"成功构建知识图谱: {document_name}, "
                f"chunks: {len(text_chunks)}, "
                f"workspace: {self.collection_id}"
            )
            
        except Exception as e:
            logger.error(f"知识图谱构建失败: {document_name}, 错误: {str(e)}")
            raise Exception(f"图谱构建失败: {str(e)}")


async def process_uploaded_document(
    document_name: str,
    oss_bucket: str,
    oss_key: str,
    collection_id: str,
    file_type: str = "md",
    source_url: Optional[str] = None,
    vectorize_only: bool = False,
    graph_only: bool = False
) -> Dict[str, Any]:
    """处理上传的文档（独立函数，供外部调用）
    
    Args:
        document_name: 文档名称
        oss_bucket: OSS 存储桶
        oss_key: OSS 对象键
        collection_id: 知识库 collection_id
        file_type: 文件类型
        vectorize_only: 仅向量化（不做图谱）
        graph_only: 仅图谱化（不做向量）
        
    Returns:
        Dict: 处理结果
    """
    from backend.config.embedding import get_embedding_model
    
    logger.info(f"process_uploaded_document 调用：vectorize_only={vectorize_only}, graph_only={graph_only}")
    
    # 根据模式初始化存储
    milvus_storage = None
    lightrag_storage = None
    
    # 向量化需要 Milvus：仅当 graph_only=False 时
    if not graph_only:
        logger.info(f"初始化 Milvus 存储：collection={collection_id}")
        milvus_storage = MilvusStorage(
            embedding_function=get_embedding_model(),
            collection_name=collection_id,
        )
    
    # 图谱化需要 LightRAG：仅当 vectorize_only=False 时
    if not vectorize_only:
        logger.info(f"初始化 LightRAG 存储：workspace={collection_id}")
        lightrag_storage = LightRAGStorage(workspace=collection_id)
    
    # 创建处理器并执行
    processor = DocumentProcessor(
        collection_id=collection_id,
        milvus_storage=milvus_storage,
        lightrag_storage=lightrag_storage
    )
    
    # 根据模式调用不同的方法
    if vectorize_only:
        logger.info(f"仅执行向量化: {document_name}")
        return await processor.vectorize_document(
            document_name=document_name,
            oss_bucket=oss_bucket,
            oss_key=oss_key,
            file_type=file_type,
            source_url=source_url
        )
    elif graph_only:
        logger.info(f"仅执行图谱化: {document_name}")
        return await processor.graph_document(
            document_name=document_name,
            oss_bucket=oss_bucket,
            oss_key=oss_key,
            file_type=file_type,
            source_url=source_url
        )
    else:
        logger.info(f"执行完整处理（向量+图谱）: {document_name}")
        # 默认：同时执行向量化和图谱化
        return await processor.process_document(
            document_name=document_name,
            oss_bucket=oss_bucket,
            oss_key=oss_key,
            file_type=file_type,
            source_url=source_url
        )
