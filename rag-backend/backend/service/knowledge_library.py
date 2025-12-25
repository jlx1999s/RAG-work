#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库服务层
提供知识库相关的业务逻辑处理
"""
import uuid
import time
import requests
import asyncio
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from backend.model.knowledge_library import KnowledgeLibrary, KnowledgeDocument
from backend.param.knowledge_library import (
    CreateLibraryRequest, UpdateLibraryRequest, AddDocumentRequest, UpdateDocumentRequest
)
from backend.param.common import Response
from backend.config.log import get_logger
from backend.config.database import DatabaseFactory

logger = get_logger(__name__)

# 全局并发控制：最多同时处理 3 个文档
_processing_semaphore = asyncio.Semaphore(3)
_processing_queue = asyncio.Queue()
_queue_processor_started = False


async def _queue_processor():
    """后台任务队列处理器，确保有序处理文档"""
    logger.info("文档处理队列处理器已启动")
    while True:
        try:
            # 从队列中获取任务
            task_data = await _processing_queue.get()
            
            if task_data is None:  # 结束信号
                break
            
            url = task_data['url']
            collection_id = task_data['collection_id']
            document_name = task_data['document_name']
            document_id = task_data.get('document_id')  # 获取文档ID
            
            logger.info(f"队列处理器开始处理文档: {document_name}, 队列剩余: {_processing_queue.qsize()}")
            
            # 使用信号量控制并发
            async with _processing_semaphore:
                await _process_uploaded_file(url, collection_id, document_name, document_id)
            
            # 标记任务完成
            _processing_queue.task_done()
            
        except Exception as e:
            logger.error(f"队列处理器异常: {str(e)}")
            _processing_queue.task_done()


async def _ensure_queue_processor():
    """确保队列处理器已启动"""
    global _queue_processor_started
    if not _queue_processor_started:
        asyncio.create_task(_queue_processor())
        _queue_processor_started = True
        logger.info("文档处理队列处理器已创建")


async def get_user_libraries(user_id: str) -> Response:
    """获取用户的知识库列表"""
    db = None
    try:
        logger.info(f"开始获取用户 {user_id} 的知识库列表")
        db = DatabaseFactory.create_session()
        
        libraries = db.query(KnowledgeLibrary).filter(
            KnowledgeLibrary.user_id == user_id,
            KnowledgeLibrary.is_active == True
        ).order_by(KnowledgeLibrary.updated_at.desc()).all()
        
        result = []
        for library in libraries:
            library_dict = library.to_dict()
            # 添加文档数量统计
            library_dict['document_count'] = len(library.documents) if library.documents else 0
            result.append(library_dict)
        
        logger.info(f"成功获取用户 {user_id} 的知识库列表，共 {len(result)} 个")
        return Response.success(result)
        
    except Exception as e:
        logger.error(f"获取用户知识库列表失败: {str(e)}")
        return Response.error(f"获取知识库列表失败: {str(e)}")
    finally:
        if db:
            db.close()


async def get_library_detail(library_id: int, user_id: str) -> Response:
    """获取知识库详情"""
    try:
        db_factory = DatabaseFactory()
        session = db_factory.create_session()
        
        try:
            library = session.query(KnowledgeLibrary).filter(
                KnowledgeLibrary.id == library_id,
                KnowledgeLibrary.user_id == user_id,
                KnowledgeLibrary.is_active == True
            ).first()
            
            if not library:
                return Response.error("知识库不存在或无权限访问")
            
            logger.info(f"成功获取知识库详情: {library.title}")
            return Response.success(library.to_dict())
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"获取知识库详情失败: {str(e)}")
        return Response.error(f"获取知识库详情失败: {str(e)}")


async def create_library(request: CreateLibraryRequest, user_id: str) -> Response:
    """创建知识库"""
    try:
        db_factory = DatabaseFactory()
        session = db_factory.create_session()
        
        try:
            # 检查同名知识库
            existing = session.query(KnowledgeLibrary).filter(
                KnowledgeLibrary.title == request.title,
                KnowledgeLibrary.user_id == user_id,
                KnowledgeLibrary.is_active == True
            ).first()
            
            if existing:
                return Response.error("已存在同名知识库")
            
            # 创建新知识库
            library = KnowledgeLibrary(
                title=request.title,
                description=request.description,
                user_id=user_id
            )
            
            session.add(library)
            session.commit()
            session.refresh(library)
            
            # 生成collection_id: kb + 知识库ID + 下划线 + 时间戳
            timestamp = str(int(time.time() * 1000))  # 毫秒级时间戳
            collection_id = f"kb{library.id}_{timestamp}"
            
            # 更新collection_id
            library.collection_id = collection_id
            session.commit()
            session.refresh(library)
            
            logger.info(f"成功创建知识库: {library.title}")
            return Response.success(library.to_dict())
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"创建知识库失败: {str(e)}")
        return Response.error(f"创建知识库失败: {str(e)}")


async def update_library(library_id: int, request: UpdateLibraryRequest, user_id: str) -> Response:
    """更新知识库"""
    try:
        db_factory = DatabaseFactory()
        session = db_factory.create_session()
        
        try:
            library = session.query(KnowledgeLibrary).filter(
                KnowledgeLibrary.id == library_id,
                KnowledgeLibrary.user_id == user_id,
                KnowledgeLibrary.is_active == True
            ).first()
            
            if not library:
                return Response.error("知识库不存在或无权限访问")
            
            # 更新字段
            if request.title is not None:
                # 检查同名知识库（排除当前库）
                existing = session.query(KnowledgeLibrary).filter(
                    KnowledgeLibrary.title == request.title,
                    KnowledgeLibrary.user_id == user_id,
                    KnowledgeLibrary.id != library_id,
                    KnowledgeLibrary.is_active == True
                ).first()
                
                if existing:
                    return Response.error("已存在同名知识库")
                
                library.title = request.title
            
            if request.description is not None:
                library.description = request.description
            
            session.commit()
            session.refresh(library)
            
            logger.info(f"成功更新知识库: {library.title}")
            return Response.success(library.to_dict())
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"更新知识库失败: {str(e)}")
        return Response.error(f"更新知识库失败: {str(e)}")


async def delete_library(library_id: int, user_id: str) -> Response:
    """删除知识库（同时清理所有向量和图谱数据）"""
    try:
        db_factory = DatabaseFactory()
        session = db_factory.create_session()
        
        try:
            library = session.query(KnowledgeLibrary).filter(
                KnowledgeLibrary.id == library_id,
                KnowledgeLibrary.user_id == user_id,
                KnowledgeLibrary.is_active == True
            ).first()
            
            if not library:
                return Response.error("知识库不存在或无权限访问")
            
            collection_id = library.collection_id
            library_title = library.title
            
            # 软删除
            library.is_active = False
            session.commit()
            
            # 异步清理 Milvus 和 Neo4j 数据
            import asyncio
            asyncio.create_task(_cleanup_library_data(
                collection_id=collection_id,
                library_title=library_title
            ))
            
            logger.info(f"成功删除知识库: {library_title}，数据清理任务已启动")
            return Response.success({"message": "知识库删除成功，图谱数据正在清理..."})
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"删除知识库失败: {str(e)}")
        return Response.error(f"删除知识库失败: {str(e)}")


async def add_document(request: AddDocumentRequest, user_id: str) -> Response:
    """添加文档到知识库"""
    try:
        db_factory = DatabaseFactory()
        session = db_factory.create_session()
        
        try:
            # 验证知识库权限
            library = session.query(KnowledgeLibrary).filter(
                KnowledgeLibrary.id == request.library_id,
                KnowledgeLibrary.user_id == user_id,
                KnowledgeLibrary.is_active == True
            ).first()
            
            if not library:
                return Response.error("知识库不存在或无权限访问")
            
            # 创建文档
            document = KnowledgeDocument(
                library_id=request.library_id,
                name=request.name,
                type=request.type,
                url=request.url,
                file_path=request.file_path,
                file_size=request.file_size
            )
            
            session.add(document)
            session.commit()
            session.refresh(document)
            
            logger.info(f"成功添加文档到知识库 {library.title}: {document.name}")
            
            # 如果是文件类型且有 URL，触发向量化和图谱构建
            if request.type == "file" and request.url:
                logger.info(f"检测到文件上传，加入处理队列: {request.url}")
                
                # 确保队列处理器已启动
                await _ensure_queue_processor()
                
                # 将任务加入队列（不阻塞）
                await _processing_queue.put({
                    'url': request.url,
                    'collection_id': library.collection_id,
                    'document_name': request.name,
                    'document_id': document.id  # 传递文档ID
                })
                
                queue_size = _processing_queue.qsize()
                logger.info(f"文档已加入处理队列，当前队列长度: {queue_size}")
            
            return Response.success(document.to_dict())
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"添加文档失败: {str(e)}")
        return Response.error(f"添加文档失败: {str(e)}")


async def update_document(document_id: int, request: UpdateDocumentRequest, user_id: str) -> Response:
    """更新文档"""
    try:
        db_factory = DatabaseFactory()
        session = db_factory.create_session()
        
        try:
            # 查询文档并验证权限
            document = session.query(KnowledgeDocument).join(KnowledgeLibrary).filter(
                KnowledgeDocument.id == document_id,
                KnowledgeLibrary.user_id == user_id,
                KnowledgeLibrary.is_active == True
            ).first()
            
            if not document:
                return Response.error("文档不存在或无权限访问")
            
            # 更新字段
            if request.name is not None:
                document.name = request.name
            if request.type is not None:
                document.type = request.type
            if request.url is not None:
                document.url = request.url
            if request.file_path is not None:
                document.file_path = request.file_path
            if request.file_size is not None:
                document.file_size = request.file_size
            
            session.commit()
            session.refresh(document)
            
            logger.info(f"成功更新文档: {document.name}")
            return Response.success(document.to_dict())
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"更新文档失败: {str(e)}")
        return Response.error(f"更新文档失败: {str(e)}")


async def delete_document(document_id: int, user_id: str) -> Response:
    """删除文档（同时清理向量数据库和图谱数据）"""
    try:
        db_factory = DatabaseFactory()
        session = db_factory.create_session()
        
        try:
            # 查询文档并验证权限
            document = session.query(KnowledgeDocument).join(KnowledgeLibrary).filter(
                KnowledgeDocument.id == document_id,
                KnowledgeLibrary.user_id == user_id,
                KnowledgeLibrary.is_active == True
            ).first()
            
            if not document:
                return Response.error("文档不存在或无权限访问")
            
            # 获取知识库信息
            library = document.library
            collection_id = library.collection_id
            document_name = document.name
            
            # 1. 删除 MySQL 记录
            session.delete(document)
            session.commit()
            logger.info(f"已删除文档记录: {document_name}")
            
            # 2. 异步清理 Milvus 和 Neo4j 数据
            import asyncio
            asyncio.create_task(_cleanup_document_data(
                collection_id=collection_id,
                document_name=document_name,
                document_id=document_id
            ))
            
            logger.info(f"成功删除文档: {document_name}，数据清理任务已启动")
            return Response.success({"message": "文档删除成功，图谱数据正在清理..."})
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        return Response.error(f"删除文档失败: {str(e)}")


async def get_processing_queue_status() -> Response:
    """获取文档处理队列状态
    
    Returns:
        Response: 包含队列长度、并发数等信息
    """
    try:
        queue_size = _processing_queue.qsize()
        available_slots = _processing_semaphore._value  # 可用并发槽位
        max_concurrent = 3  # 与 _processing_semaphore 初始化值一致
        processing_count = max_concurrent - available_slots
        
        status = {
            "queue_size": queue_size,  # 排队中的任务数
            "processing_count": processing_count,  # 正在处理的任务数
            "max_concurrent": max_concurrent,  # 最大并发数
            "available_slots": available_slots,  # 剩余可用槽位
            "processor_running": _queue_processor_started  # 处理器是否运行
        }
        
        logger.info(f"队列状态: {status}")
        return Response.success(status)
        
    except Exception as e:
        logger.error(f"获取队列状态失败: {str(e)}")
        return Response.error(f"获取队列状态失败: {str(e)}")


async def _process_uploaded_file(url: str, collection_id: str, document_name: str, document_id: int = None):
    """处理上传的文件（向量化和图谱构建）
    
    Args:
        url: 文件的 OSS URL
        collection_id: 知识库的 collection_id
        document_name: 文档名称
        document_id: 文档ID（用于更新状态）
    """
    try:
        from backend.rag.storage.milvus_storage import MilvusStorage
        from backend.rag.storage.lightrag_storage import LightRAGStorage
        from backend.config.embedding import get_embedding_model
        from backend.service.crawl import handle_md, init_crawl_status, update_crawl_status
        from backend.service.crawl import CRAWL_STATUS_PROCESSING, CRAWL_STATUS_COMPLETED, CRAWL_STATUS_ERROR
        
        logger.info(f"开始处理上传文件: {document_name}, URL: {url}")
        
        # 初始化存储
        milvus_storage = MilvusStorage(
            embedding_function=get_embedding_model(),
            collection_name=collection_id,
        )
        lightrag_storage = LightRAGStorage(workspace=collection_id)
        
        # 初始化爬虫状态（复用爬虫状态机制）
        await init_crawl_status(collection_id)
        
        # 检测文件类型
        file_like_suffixes = (".md", ".markdown", ".txt", ".pdf", ".doc", ".docx")
        url_lower = url.lower()
        
        # 对于 md/txt 文件，直接下载并处理
        if url_lower.endswith((".md", ".markdown", ".txt")):
            try:
                logger.info(f"检测到文本文件，直接下载并处理: {url}")
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                content = resp.text
                
                if not content or not content.strip():
                    msg = f"文件内容为空，跳过处理: {url}"
                    logger.warning(msg)
                    await update_crawl_status(collection_id, CRAWL_STATUS_ERROR, msg)
                    return
                
                # 调用 handle_md 进行向量化和图谱构建
                await handle_md(
                    md_content=content,
                    type="light_and_milvus",
                    param=[milvus_storage, lightrag_storage],
                    collection_id=collection_id,
                )
                
                await update_crawl_status(collection_id, CRAWL_STATUS_COMPLETED)
                
                # 标记文档为已处理
                if document_id:
                    await _mark_document_processed(document_id)
                
                logger.info(f"文件处理完成: {document_name}")
                
            except Exception as e:
                msg = f"处理文件时发生错误: {str(e)}"
                logger.error(msg)
                await update_crawl_status(collection_id, CRAWL_STATUS_ERROR, msg)
        
        # 对于其他类型文件（PDF, DOC等），暂时记录警告
        elif url_lower.endswith((".pdf", ".doc", ".docx")):
            msg = f"暂不支持自动处理 {url_lower.split('.')[-1].upper()} 格式文件，请使用网站链接方式添加"
            logger.warning(msg)
            await update_crawl_status(collection_id, CRAWL_STATUS_ERROR, msg)
        else:
            msg = f"未知文件类型: {url}"
            logger.warning(msg)
            await update_crawl_status(collection_id, CRAWL_STATUS_ERROR, msg)
            
    except Exception as e:
        logger.error(f"处理上传文件异常: {str(e)}")
        try:
            from backend.service.crawl import update_crawl_status, CRAWL_STATUS_ERROR
            await update_crawl_status(collection_id, CRAWL_STATUS_ERROR, str(e))
        except:
            pass


async def _cleanup_library_data(collection_id: str, library_title: str):
    """清理知识库相关的所有向量和图谱数据
    
    Args:
        collection_id: 知识库 collection_id
        library_title: 知识库名称
    """
    try:
        logger.info(f"开始清理知识库数据: {library_title} (collection_id: {collection_id})")
        
        # 1. 清理 Milvus collection
        try:
            from backend.rag.storage.milvus_storage import MilvusStorage
            from backend.config.embedding import get_embedding_model
            
            milvus_storage = MilvusStorage(
                embedding_function=get_embedding_model(),
                collection_name=collection_id,
            )
            
            # 删除整个 collection
            result = milvus_storage.drop_collection()
            logger.info(f"Milvus collection 删除结果: {result}")
            
        except Exception as e:
            logger.error(f"Milvus 数据清理失败: {str(e)}")
        
        # 2. 清理 LightRAG workspace（包括 Neo4j 图谱）
        try:
            from backend.rag.storage.lightrag_storage import LightRAGStorage
            
            lightrag_storage = LightRAGStorage(workspace=collection_id)
            await lightrag_storage.initialize()
            await lightrag_storage.drop_workspace()
            
            logger.info(f"LightRAG workspace 删除成功: {collection_id}")
            
        except Exception as e:
            logger.error(f"LightRAG 数据清理失败: {str(e)}")
        
        logger.info(f"知识库数据清理完成: {library_title}")
        
    except Exception as e:
        logger.error(f"清理知识库数据异常: {str(e)}")


async def _cleanup_document_data(collection_id: str, document_name: str, document_id: int):
    """清理文档相关的向量和图谱数据
    
    Args:
        collection_id: 知识库 collection_id
        document_name: 文档名称
        document_id: 文档ID
    """
    try:
        logger.info(f"开始清理文档数据: {document_name} (collection_id: {collection_id})")
        
        # 1. 清理 Milvus 向量数据
        try:
            from backend.rag.storage.milvus_storage import MilvusStorage
            from backend.config.embedding import get_embedding_model
            
            milvus_storage = MilvusStorage(
                embedding_function=get_embedding_model(),
                collection_name=collection_id,
            )
            
            # Milvus 按文档名称删除
            result = milvus_storage.delete_document(document_name)
            logger.info(f"Milvus 数据清理结果: {result}")
            
        except Exception as e:
            logger.error(f"Milvus 数据清理失败: {str(e)}")
        
        # 2. 清理 LightRAG/Neo4j 图谱数据
        # 注意：LightRAG 不支持单文档删除，只能删除整个 workspace
        # 这里记录警告，用户需要删除整个知识库才会清理图谱
        logger.warning(
            f"LightRAG 不支持单文档删除，图谱数据保留。"
            f"如需完全清理，请删除整个知识库。"
        )
        
        logger.info(f"文档数据清理完成: {document_name}")
        
    except Exception as e:
        logger.error(f"清理文档数据异常: {str(e)}")


async def _mark_document_processed(document_id: int):
    """标记文档为已处理
    
    Args:
        document_id: 文档ID
    """
    try:
        db_factory = DatabaseFactory()
        session = db_factory.create_session()
        
        try:
            document = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.id == document_id
            ).first()
            
            if document:
                document.is_processed = True
                session.commit()
                logger.info(f"文档 {document_id} 已标记为已处理")
            else:
                logger.warning(f"未找到文档 {document_id}")
                
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"标记文档已处理失败: {str(e)}")


async def get_document_content(document_id: int, user_id: str) -> Response:
    """获取文档内容，用于预览

    目前主要支持：
    - type = "file" 且 url 指向可直接下载的 md/txt 文本
    - 其他类型暂时只返回基础信息
    """
    try:
        db_factory = DatabaseFactory()
        session = db_factory.create_session()

        try:
            # 查询文档并验证权限
            document = (
                session.query(KnowledgeDocument)
                .join(KnowledgeLibrary)
                .filter(
                    KnowledgeDocument.id == document_id,
                    KnowledgeLibrary.user_id == user_id,
                    KnowledgeLibrary.is_active == True,
                )
                .first()
            )

            if not document:
                return Response.error("文档不存在或无权限访问")

            data: Dict[str, Any] = {
                "id": document.id,
                "library_id": document.library_id,
                "name": document.name,
                "type": document.type,
                "url": document.url,
            }

            # 优先处理可直接下载的文本类文件
            url = document.url or ""
            lower_url = url.lower()

            if document.type in ("file", "link") and lower_url.endswith(
                (".md", ".markdown", ".txt")
            ):
                try:
                    logger.info(f"从URL获取文档内容: {url}")
                    resp = requests.get(url, timeout=30)
                    resp.raise_for_status()
                    content_text = resp.text or ""

                    if not content_text.strip():
                        return Response.error("文档内容为空，无法预览")

                    data["content"] = content_text
                    # 简单标记内容类型，前端可以按需要处理
                    data["content_type"] = (
                        "text/markdown" if lower_url.endswith((".md", ".markdown")) else "text/plain"
                    )

                    return Response.success(data)
                except Exception as e:
                    logger.error(f"从URL获取文档内容失败: {str(e)}")
                    return Response.error(f"获取文档内容失败: {str(e)}")

            # 对于暂不支持直接预览的类型，返回基本信息
            logger.info(
                f"文档 {document.id} 类型为 {document.type}，当前暂不支持直接内容预览，只返回元信息"
            )
            return Response.success(data)

        finally:
            session.close()

    except Exception as e:
        logger.error(f"获取文档内容失败: {str(e)}")
        return Response.error(f"获取文档内容失败: {str(e)}")