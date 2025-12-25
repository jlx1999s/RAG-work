from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from backend.param.knowledge_library import (
    CreateLibraryRequest, UpdateLibraryRequest, AddDocumentRequest, UpdateDocumentRequest
)
from backend.param.crawl import UploadDocRequest
from backend.param.common import Response
from backend.service import knowledge_library as library_service
from backend.config.log import get_logger
from backend.config.dependencies import get_current_user
import os
import time

logger = get_logger(__name__)

router = APIRouter(
    prefix="/knowledge",
    tags=["KNOWLEDGE_LIBRARY"]
)


@router.get("/libraries")
async def get_libraries(current_user: int = Depends(get_current_user)):
    """获取用户的知识库列表"""
    logger.info(f"用户 {current_user} 请求获取知识库列表")
    return await library_service.get_user_libraries(current_user)


@router.get("/libraries/{library_id}")
async def get_library(library_id: int, current_user: int = Depends(get_current_user)):
    """获取知识库详情"""
    logger.info(f"用户 {current_user} 请求获取知识库详情: {library_id}")
    return await library_service.get_library_detail(library_id, current_user)


@router.post("/libraries")
async def create_library(request: CreateLibraryRequest, current_user: int = Depends(get_current_user)):
    """创建知识库"""
    logger.info(f"用户 {current_user} 请求创建知识库: {request.title}")
    return await library_service.create_library(request, current_user)


@router.put("/libraries/{library_id}")
async def update_library(
    library_id: int,
    request: UpdateLibraryRequest,
    current_user: int = Depends(get_current_user)
):
    """更新知识库"""
    logger.info(f"用户 {current_user} 请求更新知识库: {library_id}")
    return await library_service.update_library(library_id, request, current_user)


@router.delete("/libraries/{library_id}")
async def delete_library(library_id: int, current_user: int = Depends(get_current_user)):
    """删除知识库"""
    logger.info(f"用户 {current_user} 请求删除知识库: {library_id}")
    return await library_service.delete_library(library_id, current_user)


@router.post("/documents")
async def add_document(
    request: AddDocumentRequest,
    current_user: int = Depends(get_current_user)
):
    """添加文档到知识库"""
    logger.info(f"用户 {current_user} 请求添加文档到知识库: {request.library_id}")
    return await library_service.add_document(request, current_user)


@router.put("/documents/{document_id}")
async def update_document(
    document_id: int,
    request: UpdateDocumentRequest,
    current_user: int = Depends(get_current_user)
):
    """更新文档"""
    logger.info(f"用户 {current_user} 请求更新文档: {document_id}")
    return await library_service.update_document(document_id, request, current_user)


@router.delete("/documents/{document_id}")
async def delete_document(document_id: int, current_user: int = Depends(get_current_user)):
    """删除文档"""
    logger.info(f"用户 {current_user} 请求删除文档: {document_id}")
    return await library_service.delete_document(document_id, current_user)


@router.post("/documents/{document_id}/process")
async def start_document_processing(document_id: int, current_user: int = Depends(get_current_user)):
    """开始解析文档（用户点击解析按钮）"""
    logger.info(f"用户 {current_user} 请求开始解析文档: {document_id}")
    return await library_service.start_document_processing(document_id, current_user)


@router.post("/documents/{document_id}/vectorize")
async def start_document_vectorize(document_id: int, current_user: int = Depends(get_current_user)):
    """仅向量化文档（不做图谱）"""
    logger.info(f"用户 {current_user} 请求向量化文档: {document_id}")
    return await library_service.start_document_vectorize(document_id, current_user)


@router.post("/documents/{document_id}/graph")
async def start_document_graph(document_id: int, current_user: int = Depends(get_current_user)):
    """仅图谱化文档（不做向量）"""
    logger.info(f"用户 {current_user} 请求图谱化文档: {document_id}")
    return await library_service.start_document_graph(document_id, current_user)


@router.get("/documents/{document_id}/content")
async def get_document_content(document_id: int, current_user: int = Depends(get_current_user)):
    """获取文档内容用于预览（支持 md/txt 等纯文本）"""
    logger.info(f"用户 {current_user} 请求获取文档内容: {document_id}")
    return await library_service.get_document_content(document_id, current_user)


@router.get("/documents/{document_id}/chunks")
async def get_document_chunks(document_id: int, current_user: int = Depends(get_current_user)):
    """获取文档的分块信息"""
    logger.info(f"用户 {current_user} 请求获取文档分块信息: {document_id}")
    return await library_service.get_document_chunks(document_id, current_user)


@router.get("/processing/queue-status")
async def get_queue_status(current_user: int = Depends(get_current_user)):
    """获取文档处理队列状态"""
    logger.info(f"用户 {current_user} 请求获取队列状态")
    return await library_service.get_processing_queue_status()


@router.post("/upload-url")
async def get_upload_url(
    request: UploadDocRequest,
    current_user: int = Depends(get_current_user)
):
    """获取文件上传的预签名URL"""
    try:
        logger.info(f"用户 {current_user} 请求获取上传URL: {request.document_name}")
        if not request.document_name:
            raise HTTPException(status_code=400, detail="document_name不能为空")

        from backend.config.oss import get_presigned_url_for_upload

        bucket = os.getenv("OSS_BUCKET_FILE", os.getenv("OSS_BUCKET_NAME", "ragagent-file"))
        key = f"user_{current_user}/{int(time.time())}_{request.document_name}"
        presign = get_presigned_url_for_upload(bucket=bucket, key=key)

        return Response.success({
            "url": presign.get("url"),
            "method": presign.get("method", "PUT"),
            "signed_headers": presign.get("signed_headers", {})
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取上传URL失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取上传URL失败: {str(e)}")
