from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from backend.param.chat import ChatRequest, CreateConversationRequest, UpdateConversationTitleRequest
from backend.param.common import Response
from backend.service import chat as chat_service
from backend.service import conversation as conversation_service
from backend.config.log import get_logger
from backend.config.dependencies import get_current_user
import json
import os
import time
from typing import Optional

logger = get_logger(__name__)

router = APIRouter(
    prefix="/llm",
    tags=["CHAT"]
)


def _ensure_user_match(target_user_id: str | int, current_user: int) -> None:
    if str(target_user_id) != str(current_user):
        raise HTTPException(status_code=403, detail="无权限访问其他用户资源")


async def _ensure_conversation_owner(conversation_id: str, current_user: int) -> None:
    conv_result = await conversation_service.get_conversation_by_id(conversation_id)
    if not conv_result.get("success"):
        raise HTTPException(status_code=404, detail="对话不存在")
    conv_user_id = str((conv_result.get("data") or {}).get("user_id") or "")
    if conv_user_id != str(current_user):
        raise HTTPException(status_code=403, detail="无权限访问该会话")

@router.post('/chat')
async def chat(
    chat_param: ChatRequest, 
    current_user: int = Depends(get_current_user)
):
    """
    聊天接口 - 非流式响应
    
    Args:
        chat_param: 聊天请求参数
        current_user: 当前用户邮箱
        
    Returns:
        聊天响应结果
    """
    try:
        logger.info(f"用户 {current_user} 发起聊天请求: {chat_param.content}")
        
        # 验证请求参数
        if not chat_param.content:
            raise HTTPException(status_code=400, detail="聊天内容不能为空")
        
        # 强制使用鉴权用户，防止前端伪造 user_id
        chat_param.user_id = str(current_user)

        # 调用服务层处理聊天
        result = await chat_service.chat(chat_param)
        
        if result.get("success"):
            logger.info("聊天请求处理成功")
            return result
        else:
            logger.error(f"聊天请求处理失败: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result.get("message", "聊天处理失败"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"聊天接口异常: {str(e)}")
        logger.exception("详细错误信息:")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.post('/chat/stream')
async def chat_stream(
    chat_param: ChatRequest,
    current_user: int = Depends(get_current_user)
):
    """
    聊天接口 - 流式响应
    
    Args:
        chat_param: 聊天请求参数
        current_user: 当前用户邮箱
        
    Returns:
        StreamingResponse: 流式聊天响应
    """
    try:
        logger.info(f"用户 {current_user} 发起流式聊天请求: {chat_param.content}")
        
        # 验证请求参数
        if not chat_param.content:
            raise HTTPException(status_code=400, detail="聊天内容不能为空")
        
        # 强制使用鉴权用户，防止前端伪造 user_id
        chat_param.user_id = str(current_user)

        async def generate_response():
            """生成流式响应数据"""
            try:
                async for chunk in chat_service.chat_stream(chat_param):
                    # 将数据转换为 JSON 格式并添加换行符
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error(f"流式响应生成失败: {str(e)}")
                error_chunk = {
                    "type": "error",
                    "error": str(e),
                    "message": "流式响应生成失败"
                }
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"流式聊天接口异常: {str(e)}")
        logger.exception("详细错误信息:")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.get('/history/{user_id}')
async def get_chat_history(
    user_id: str,
    conversation_id: Optional[str] = None,
    current_user: int = Depends(get_current_user)
):
    """
    获取聊天历史
    
    Args:
        user_id: 用户ID
        conversation_id: 会话ID（可选）
        current_user: 当前用户邮箱
        
    Returns:
        聊天历史数据
    """
    try:
        logger.info(f"用户 {current_user} 获取用户 {user_id} 的聊天历史")
        
        _ensure_user_match(user_id, current_user)
        result = await chat_service.get_chat_history_list(str(current_user), conversation_id)
        
        if result.get("success"):
            return Response.success(result)
        else:
            raise HTTPException(status_code=500, detail=result.get("message", "获取聊天历史失败"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取聊天历史接口异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.get('/history/single/{conversation_id}')
async def get_single_conversation_history(
    conversation_id: str,
    current_user: int = Depends(get_current_user)
) -> Response:
    """
    查询一轮对话历史记录的接口
    
    Args:
        conversation_id: 会话ID
        current_user: 当前用户邮箱
        
    Returns:
        Response: 包含单轮对话历史记录的响应
    """
    try:
        logger.info(f"用户 {current_user} 查询会话 {conversation_id} 的历史记录")
        
        await _ensure_conversation_owner(conversation_id, current_user)

        # 调用服务层获取单轮对话历史
        result = await chat_service.get_chat_history_list(str(current_user), conversation_id)
        
        if result.get("success"):
            return Response.success(result)
        else:
            return Response.error(result.get("message", "获取对话历史失败"))
            
    except Exception as e:
        logger.error(f"查询单轮对话历史接口异常: {str(e)}")
        return Response.error(f"服务器内部错误: {str(e)}")


@router.get('/history/titles/{user_id}')
async def get_chat_history_titles(
    user_id: str,
    current_user: int = Depends(get_current_user)
) -> Response:
    """
    查询所有历史记录标题列表的接口
    
    Args:
        user_id: 用户ID
        current_user: 当前用户邮箱
        
    Returns:
        Response: 包含所有历史记录标题列表的响应
    """
    try:
        logger.info(f"用户 {current_user} 查询用户 {user_id} 的所有历史记录标题")
        
        _ensure_user_match(user_id, current_user)

        # 调用服务层获取用户的所有会话列表
        result = await chat_service.get_user_conversations(str(current_user))
        
        if result.get("success"):
            return Response.success(result.get("data"))
        else:
            return Response.error(result.get("message", "获取会话列表失败"))
        
    except Exception as e:
        logger.error(f"查询历史记录标题列表接口异常: {str(e)}")
        return Response.error(f"服务器内部错误: {str(e)}")


@router.post('/conversation')
async def create_conversation(
    request: CreateConversationRequest,
    current_user: int = Depends(get_current_user)
) -> Response:
    """
    创建新的会话
    
    Args:
        request: 创建会话请求体
        current_user: 当前用户邮箱
        
    Returns:
        Response: 包含新创建会话信息的响应
    """
    try:
        _ensure_user_match(request.user_id, current_user)
        logger.info(f"用户 {current_user} 创建新会话")
        
        result = await chat_service.create_conversation(str(current_user), request.title)
        
        if result.get("success"):
            return Response.success(result.get("data"))
        else:
            return Response.error(result.get("message", "创建会话失败"))
            
    except Exception as e:
        logger.error(f"创建会话接口异常: {str(e)}")
        return Response.error(f"服务器内部错误: {str(e)}")


@router.put('/conversation/{conversation_id}/title')
async def update_conversation_title(
    conversation_id: str,
    request: UpdateConversationTitleRequest,
    current_user: int = Depends(get_current_user)
) -> Response:
    """
    更新会话标题
    
    Args:
        conversation_id: 会话ID
        request: 更新标题请求体
        current_user: 当前用户邮箱
        
    Returns:
        Response: 更新结果响应
    """
    try:
        await _ensure_conversation_owner(conversation_id, current_user)
        logger.info(f"用户 {current_user} 更新会话 {conversation_id} 的标题为: {request.title}")
        
        result = await chat_service.update_conversation_title(conversation_id, request.title)
        
        if result.get("success"):
            return Response.success(result.get("data"))
        else:
            return Response.error(result.get("message", "更新会话标题失败"))
            
    except Exception as e:
        logger.error(f"更新会话标题接口异常: {str(e)}")
        return Response.error(f"服务器内部错误: {str(e)}")


@router.delete('/conversation/{conversation_id}')
async def delete_conversation(
    conversation_id: str,
    current_user: int = Depends(get_current_user)
) -> Response:
    """
    删除会话
    
    Args:
        conversation_id: 会话ID
        current_user: 当前用户邮箱
        
    Returns:
        Response: 删除结果响应
    """
    try:
        await _ensure_conversation_owner(conversation_id, current_user)
        logger.info(f"用户 {current_user} 删除会话 {conversation_id}")
        
        result = await chat_service.delete_conversation(conversation_id)
        
        if result.get("success"):
            return Response.success(result.get("data"))
        else:
            return Response.error(result.get("message", "删除会话失败"))
            
    except Exception as e:
        logger.error(f"删除会话接口异常: {str(e)}")
        return Response.error(f"服务器内部错误: {str(e)}")


@router.delete('/conversations/{user_id}')
async def delete_user_conversations(
    user_id: str,
    current_user: int = Depends(get_current_user)
) -> Response:
    """
    删除用户的所有会话
    
    Args:
        user_id: 用户ID
        current_user: 当前用户邮箱
        
    Returns:
        Response: 删除结果响应
    """
    try:
        _ensure_user_match(user_id, current_user)
        logger.info(f"用户 {current_user} 删除自己的所有会话")
        
        result = await chat_service.delete_user_conversations(str(current_user))
        
        if result.get("success"):
            return Response.success(result.get("data"))
        else:
            return Response.error(result.get("message", "删除用户会话失败"))
            
    except Exception as e:
        logger.error(f"删除用户会话接口异常: {str(e)}")
        return Response.error(f"服务器内部错误: {str(e)}")


@router.get('/memory/{user_id}')
async def get_user_memory(
    user_id: str,
    collection_id: str = Query("kb12_1760260169325"),
    profile_limit: int = Query(50, ge=1, le=200),
    event_limit: int = Query(50, ge=1, le=200),
    conversation_id: Optional[str] = Query(None),
    short_term_limit: int = Query(10, ge=1, le=50),
    current_user: int = Depends(get_current_user)
) -> Response:
    try:
        _ensure_user_match(user_id, current_user)
        logger.info(f"用户 {current_user} 查询用户 {user_id} 的记忆快照")
        result = await chat_service.get_user_memory_snapshot(
            user_id=user_id,
            collection_id=collection_id,
            profile_limit=profile_limit,
            event_limit=event_limit,
            conversation_id=conversation_id,
            short_term_limit=short_term_limit
        )
        if result.get("success"):
            return Response.success(result.get("data"))
        return Response.error(result.get("message", "获取用户记忆失败"))
    except Exception as e:
        logger.error(f"查询用户记忆接口异常: {str(e)}")
        return Response.error(f"服务器内部错误: {str(e)}")


@router.delete('/memory/{user_id}/profile/{memory_key}')
async def delete_user_memory_profile(
    user_id: str,
    memory_key: str,
    collection_id: str = Query("kb12_1760260169325"),
    current_user: int = Depends(get_current_user)
) -> Response:
    try:
        _ensure_user_match(user_id, current_user)
        logger.info(f"用户 {current_user} 删除用户 {user_id} 的画像记忆 {memory_key}")
        result = await chat_service.remove_user_memory_profile(
            user_id=user_id,
            collection_id=collection_id,
            memory_key=memory_key
        )
        if result.get("success"):
            return Response.success(result.get("data"))
        return Response.error(result.get("message", "删除画像记忆失败"))
    except Exception as e:
        logger.error(f"删除画像记忆接口异常: {str(e)}")
        return Response.error(f"服务器内部错误: {str(e)}")


@router.delete('/memory/{user_id}/conversation/{conversation_id}')
async def delete_conversation_memory(
    user_id: str,
    conversation_id: str,
    collection_id: str = Query("kb12_1760260169325"),
    current_user: int = Depends(get_current_user)
) -> Response:
    try:
        _ensure_user_match(user_id, current_user)
        await _ensure_conversation_owner(conversation_id, current_user)
        logger.info(f"用户 {current_user} 删除用户 {user_id} 的会话记忆 {conversation_id}")
        result = await chat_service.remove_conversation_memory_events(
            user_id=user_id,
            collection_id=collection_id,
            conversation_id=conversation_id
        )
        if result.get("success"):
            return Response.success(result.get("data"))
        return Response.error(result.get("message", "删除会话记忆失败"))
    except Exception as e:
        logger.error(f"删除会话记忆接口异常: {str(e)}")
        return Response.error(f"服务器内部错误: {str(e)}")


@router.delete('/memory/{user_id}')
async def delete_all_user_memory(
    user_id: str,
    collection_id: str = Query("kb12_1760260169325"),
    current_user: int = Depends(get_current_user)
) -> Response:
    try:
        _ensure_user_match(user_id, current_user)
        logger.info(f"用户 {current_user} 删除用户 {user_id} 在知识库 {collection_id} 的全部记忆")
        result = await chat_service.remove_all_user_memory(
            user_id=user_id,
            collection_id=collection_id
        )
        if result.get("success"):
            return Response.success(result.get("data"))
        return Response.error(result.get("message", "删除全部记忆失败"))
    except Exception as e:
        logger.error(f"删除全部记忆接口异常: {str(e)}")
        return Response.error(f"服务器内部错误: {str(e)}")


@router.post('/get-url')
async def get_signature(
    document_name: str = Query("upload.bin", min_length=1),
    bucket: Optional[str] = Query(None),
    expire_seconds: int = Query(3600, ge=60, le=86400),
    current_user: int = Depends(get_current_user)
):
    """
    获取签名URL（原有接口保留）
    
    Returns:
        签名URL信息
    """
    try:
        from backend.config.oss import get_presigned_url_for_upload, get_presigned_url_for_download
        resolved_bucket = str(
            bucket or os.getenv("OSS_BUCKET_FILE") or os.getenv("OSS_BUCKET_NAME") or "ragagent-file"
        )
        object_key = f"user_{current_user}/{int(time.time())}_{document_name}"
        upload_presign = get_presigned_url_for_upload(
            bucket=resolved_bucket,
            key=object_key,
            expire_seconds=expire_seconds
        )
        download_presign = get_presigned_url_for_download(
            bucket=resolved_bucket,
            key=object_key,
            expire_seconds=expire_seconds
        )
        return Response.success({
            "url": upload_presign.get("url"),
            "method": upload_presign.get("method", "PUT"),
            "signed_headers": upload_presign.get("signed_headers", {}),
            "object_key": object_key,
            "bucket": resolved_bucket,
            "download_url": download_presign.get("url"),
            "expiration": upload_presign.get("expiration")
        })
    except Exception as e:
        logger.error(f"获取签名URL失败: {str(e)}")
        return Response.error(f"获取签名URL失败: {str(e)}")
