#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档处理器
专门处理用户上传的文档（与爬虫逻辑完全分离）
"""
import os
from typing import Optional, Dict, Any
from backend.config.log import get_logger
from backend.config.oss import OssClientFactory
from backend.rag.storage.milvus_storage import MilvusStorage
from backend.rag.storage.lightrag_storage import LightRAGStorage
from backend.rag.chunks.chunks import TextChunker
from backend.rag.chunks.models import ChunkConfig, ChunkStrategy, DocumentContent

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
        file_type: str = "md"
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
            content = await self._read_from_oss(oss_bucket, oss_key)
            
            if not content or not content.strip():
                raise ValueError(f"文档内容为空: {document_name}")
            
            logger.info(f"成功从 OSS 读取文档，内容长度: {len(content)} 字符")
            
            # 2. 文档分块
            chunks_result = await self._chunk_document(content, document_name, file_type)
            
            if not chunks_result or not chunks_result.chunks:
                raise ValueError(f"文档分块结果为空: {document_name}")
            
            chunk_count = len(chunks_result.chunks)
            logger.info(f"文档分块完成，共 {chunk_count} 个 chunks")
            
            # 3. 存储到 Milvus
            await self._store_to_milvus(chunks_result, document_name)
            
            # 4. 构建知识图谱（LightRAG + Neo4j）
            await self._build_knowledge_graph(chunks_result, document_name)
            
            result = {
                "status": "success",
                "document_name": document_name,
                "chunk_count": chunk_count,
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
        file_type: str = "md"
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
            content = await self._read_from_oss(oss_bucket, oss_key)
            
            if not content or not content.strip():
                raise ValueError(f"文档内容为空: {document_name}")
            
            logger.info(f"成功从 OSS 读取文档，内容长度: {len(content)} 字符")
            
            # 2. 文档分块
            chunks_result = await self._chunk_document(content, document_name, file_type)
            
            if not chunks_result or not chunks_result.chunks:
                raise ValueError(f"文档分块结果为空: {document_name}")
            
            chunk_count = len(chunks_result.chunks)
            logger.info(f"文档分块完成，共 {chunk_count} 个 chunks")
            
            # 3. 存储到 Milvus
            await self._store_to_milvus(chunks_result, document_name)
            
            result = {
                "status": "success",
                "document_name": document_name,
                "chunk_count": chunk_count,
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
        file_type: str = "md"
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
            content = await self._read_from_oss(oss_bucket, oss_key)
            
            if not content or not content.strip():
                raise ValueError(f"文档内容为空: {document_name}")
            
            logger.info(f"成功从 OSS 读取文档，内容长度: {len(content)} 字符")
            
            # 2. 文档分块
            chunks_result = await self._chunk_document(content, document_name, file_type)
            
            if not chunks_result or not chunks_result.chunks:
                raise ValueError(f"文档分块结果为空: {document_name}")
            
            chunk_count = len(chunks_result.chunks)
            logger.info(f"文档分块完成，共 {chunk_count} 个 chunks")
            
            # 3. 构建知识图谱
            await self._build_knowledge_graph(chunks_result, document_name)
            
            result = {
                "status": "success",
                "document_name": document_name,
                "chunk_count": chunk_count,
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
    
    async def _read_from_oss(self, bucket: str, key: str) -> str:
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
            
            # 读取内容并解码
            content = result.body.read().decode('utf-8')
            
            logger.info(f"成功从 OSS 读取: {bucket}/{decoded_key}, 大小: {len(content)} bytes")
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
            if file_type in ["md", "markdown"]:
                strategy = ChunkStrategy.MARKDOWN_HEADER
            else:
                strategy = ChunkStrategy.MARKDOWN_HEADER  # 默认也用 Markdown 策略
            
            chunk_config = ChunkConfig(strategy=strategy)
            
            # 创建文档对象
            document = DocumentContent(
                content=content,
                document_name=document_name
            )
            
            # 执行分块
            result = self.chunker.chunk_document(document, chunk_config)
            
            logger.info(f"文档分块完成: {document_name}, 策略: {strategy.value}")
            return result
            
        except Exception as e:
            logger.error(f"文档分块失败: {document_name}, 错误: {str(e)}")
            raise
    
    async def _store_to_milvus(self, chunks_result: Any, document_name: str):
        """将分块存储到 Milvus
        
        Args:
            chunks_result: 分块结果
            document_name: 文档名称
        """
        try:
            result = self.milvus_storage.store_chunks_batch([chunks_result])
            
            logger.info(
                f"成功存储到 Milvus: {document_name}, "
                f"chunks: {result.get('total_chunks', 0)}, "
                f"collection: {self.collection_id}"
            )
            
        except Exception as e:
            logger.error(f"Milvus 存储失败: {document_name}, 错误: {str(e)}")
            raise Exception(f"向量存储失败: {str(e)}")
    
    async def _build_knowledge_graph(self, chunks_result: Any, document_name: str):
        """构建知识图谱（LightRAG + Neo4j）
        
        Args:
            chunks_result: 分块结果
            document_name: 文档名称
        """
        try:
            # 提取所有 chunk 的文本内容
            text_chunks = [chunk.page_content for chunk in chunks_result.chunks]
            
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
            file_type=file_type
        )
    elif graph_only:
        logger.info(f"仅执行图谱化: {document_name}")
        return await processor.graph_document(
            document_name=document_name,
            oss_bucket=oss_bucket,
            oss_key=oss_key,
            file_type=file_type
        )
    else:
        logger.info(f"执行完整处理（向量+图谱）: {document_name}")
        # 默认：同时执行向量化和图谱化
        return await processor.process_document(
            document_name=document_name,
            oss_bucket=oss_bucket,
            oss_key=oss_key,
            file_type=file_type
        )
