#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聊天服务层
基于 RAGGraph 提供聊天功能
"""
import uuid
from typing import Dict, Any, AsyncGenerator, Optional, List
from backend.config.agent import get_rag_graph_for_collection
from backend.agent.contexts.raggraph_context import RAGContext
from backend.param.chat import ChatRequest
from backend.config.log import get_logger
from backend.service import conversation as conversation_service
from backend.service.chat_history import save_chat_message, get_chat_messages
from backend.agent.prompts.raggraph_prompt import RAGGraphPrompts

logger = get_logger(__name__)


def _make_serializable(obj: Any) -> Any:
    """
    将对象转换为可序列化的格式
    
    Args:
        obj: 需要序列化的对象
        
    Returns:
        Any: 可序列化的对象
    """
    if hasattr(obj, 'dict'):
        return obj.dict()
    elif hasattr(obj, '__dict__'):
        return {k: _make_serializable(v) for k, v in obj.__dict__.items() 
                if not k.startswith('_')}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)


def _validate_chat_request(chat_request: ChatRequest) -> Dict[str, Any]:
    """
    验证聊天请求参数
    
    Args:
        chat_request: 聊天请求参数
        
    Returns:
        Dict[str, Any]: 验证结果
    """
    if not chat_request:
        return {
            "valid": False,
            "error": "聊天请求不能为空"
        }
    
    if not chat_request.content or not str(chat_request.content).strip():
        return {
            "valid": False,
            "error": "聊天内容不能为空"
        }
    
    # 限制内容长度
    content = str(chat_request.content).strip()
    if len(content) > 10000:  # 10KB限制
        return {
            "valid": False,
            "error": "聊天内容过长，请控制在10000字符以内"
        }
    
    # 验证用户ID
    user_id = chat_request.user_id or "default_user"
    if not str(user_id).strip():
        return {
            "valid": False,
            "error": "用户ID不能为空"
        }
    
    return {
        "valid": True,
        "user_id": user_id,
        "content": content
    }

async def chat_stream(chat_request: ChatRequest) -> AsyncGenerator[Dict[str, Any], None]:
    """
    处理聊天请求 - 流式响应
    
    Args:
        chat_request: 聊天请求参数
        
    Yields:
        Dict[str, Any]: 流式聊天响应数据
    """
    try:
        logger.info(f"开始处理流式聊天请求: {chat_request.content[:100]}...")
        
        # 验证请求参数
        validation = _validate_chat_request(chat_request)
        if not validation["valid"]:
            yield {
                "type": "error",
                "error": validation["error"],
                "message": "聊天请求参数无效"
            }
            return
        
        user_id = validation["user_id"]
        content = validation["content"]
        
        # 获取 collection_id，如果没有提供则使用默认值
        collection_id = chat_request.collection_id or "kb12_1760260169325"
        logger.info(f"\n===========\n使用 collection_id={collection_id} 处理聊天请求\n===========\n")
        
        # 基于 collection_id 动态创建 RAGGraph 实例
        try:
            rag_graph = get_rag_graph_for_collection(collection_id)
        except Exception as e:
            logger.error(f"创建RAGGraph实例失败，collection_id={collection_id}: {str(e)}")
            yield {
                "type": "error",
                "error": f"创建RAGGraph实例失败: {str(e)}",
                "message": "聊天服务不可用"
            }
            return
        # 处理conversation_id
        session_id = chat_request.conversation_id
        
        # 如果没有提供conversation_id，创建新的对话
        if not session_id or not str(session_id).strip():
            # 创建新对话
            title = content[:50] + "..." if len(content) > 50 else content
            result = await conversation_service.create_conversation(
                user_id=user_id,
                title=title
            )
            if result.get("success"):
                session_id = result["data"]["conversation_id"]
                logger.info(f"创建新对话: {session_id}")
            else:
                logger.error(f"创建对话失败: {result.get('message')}")
                yield {
                    "type": "error",
                    "error": result.get("error", "创建对话失败"),
                    "message": "创建对话失败"
                }
                return
        else:
            # 验证对话是否存在并更新时间戳
            conv_result = await conversation_service.get_conversation_by_id(session_id)
            if not conv_result.get("success"):
                yield {
                    "type": "error",
                    "error": "对话不存在",
                    "message": "指定的对话不存在"
                }
                return
            
            # 更新现有对话的时间戳
            await conversation_service.update_conversation_timestamp(session_id)
        
        # 创建 RAG 上下文
        context = RAGContext(
            session_id=session_id,
            user_id=user_id,
            retrieval_mode=chat_request.retrieval_mode,
            max_retrieval_docs=chat_request.max_retrieval_docs or 3,
            system_prompt=chat_request.system_prompt or "你是一个专业的RAG助手，能够基于检索到的信息提供准确的回答。"
        )
        
        # ===== 构造带长短期记忆的对话历史（会话内） =====
        history_messages: List[Dict[str, str]] = []
        conversation_summary: Optional[str] = None
        try:
            # 从数据库获取该会话的历史消息（已按时间升序）
            history_records = get_chat_messages(session_id)

            dialog_records: List[Dict[str, Any]] = []
            summary_records: List[Dict[str, Any]] = []
            for record in history_records:
                role = (record.get("role") or "").lower()
                msg_type = (record.get("type") or "").lower()
                content_text = (record.get("content") or "").strip()
                if not content_text:
                    continue
                if msg_type == "messages" and role in ["user", "assistant"]:
                    dialog_records.append(record)
                elif msg_type == "summary":
                    summary_records.append(record)

            # 已存在的摘要（仅在当前 conversation_id 内生效）
            if summary_records:
                conversation_summary = summary_records[-1].get("content") or None

            # 将较早的对话压缩为摘要（对话内长期记忆），仅在消息较多且尚无摘要时触发
            if not conversation_summary and len(dialog_records) > 12:
                # 取较早的部分作为摘要来源，例如前 len(dialog_records) - 8 条
                long_records = dialog_records[:-8]
                lines: List[str] = []
                for r in long_records:
                    role = (r.get("role") or "").lower()
                    text = (r.get("content") or "").strip()
                    if not text:
                        continue
                    if role == "user":
                        prefix = "用户"
                    else:
                        prefix = "助手"
                    lines.append(f"{prefix}: {text}")
                if lines:
                    history_text = "\n".join(lines)
                    try:
                        prompt = RAGGraphPrompts.get_conversation_summary_prompt().format(history=history_text)
                        summary_result = rag_graph.llm.invoke(prompt)
                        conversation_summary = getattr(summary_result, "content", None) or str(summary_result)
                        # 将摘要写入 chat_history，type=summary，只在当前会话内使用
                        save_chat_message(
                            conversation_id=session_id,
                            role="system",
                            message_type="summary",
                            content=conversation_summary,
                            extra_data={"node_name": "conversation_summary"}
                        )
                        logger.info("已为当前会话生成对话摘要")
                    except Exception as summarise_error:
                        logger.error(f"生成对话摘要失败: {summarise_error}")
                        conversation_summary = None

            # 短期记忆：保留最近 10 条 user/assistant 对话
            short_dialog = dialog_records[-10:] if len(dialog_records) > 10 else dialog_records
            for r in short_dialog:
                role = (r.get("role") or "").lower()
                content_text = (r.get("content") or "").strip()
                history_messages.append({
                    "role": "user" if role == "user" else "assistant",
                    "content": content_text,
                })

        except Exception as e:
            logger.error(f"加载历史消息/摘要失败，将仅使用当前问题: {str(e)}")
            history_messages = []
            conversation_summary = None

        # 准备输入数据：会话摘要（如有） + 短期历史对话 + 当前这轮用户问题
        input_messages: List[Dict[str, str]] = []
        if conversation_summary:
            input_messages.append({
                "role": "system",
                "content": f"本次对话的背景摘要：{conversation_summary}",
            })
        input_messages.extend(history_messages)
        input_messages.append({"role": "user", "content": content})

        input_data = {
            "messages": input_messages
        }
        
        # 发送开始信号
        yield {
            "type": "start",
            "session_id": session_id,
            "user_id": user_id,
            "message": "开始处理聊天请求"
        }

        
        # 存储用户输入消息到数据库
        save_chat_message(
            conversation_id=session_id,
            role="user",
            message_type="messages",
            content=content,
            extra_data={"node_name": "user_input"}
        )
        
        # 调用 RAGGraph stream 方法
        logger.info("调用 RAGGraph.stream 方法...")
        
        try:
            # 使用 stream_mode="messages" 进行流式处理
            async for mode,chunk in rag_graph.astream(input_data, context, stream_mode="mix"):
                if mode == "updates":
                     # 显示节点名称
                    node_name = list(chunk.keys())[0]
                    node_output = chunk[node_name]
                    logger.info(f"（流式输出）节点名称: {node_name}")
                    #logger.info(f"节点输出: {node_output}")
                    
                    # 根据节点类型处理content
                    content = ""
                    if node_name == "check_retrieval_needed":
                        content = f"节点名称为{node_name}，LLM判断是否需要检索结果为{node_output['need_retrieval']}，理由为{node_output['need_retrieval_reason']}，提取原始问题为{node_output['original_question']}"
                    elif node_name == "expand_subquestions":
                        extraquestion = "\n".join([f"{i+1}. {q}" for i, q in enumerate(node_output['subquestions'])])
                        content = f"节点名称为{node_name}，扩展子问题为{extraquestion}"
                    elif node_name == "classify_question_type":
                        content = f"节点名称为{node_name}，LLM判断检索模式为{node_output['retrieval_mode']}，理由为{node_output['retrieval_mode_reason']}"
                    elif node_name == "vector_db_retrieval":
                        vectordoc = "\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(node_output['vector_db_results'])])
                        content = f"节点名称为{node_name}，向量检索到的文档为{vectordoc}"
                    elif node_name == "graph_db_retrieval":
                        graphdoc = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(node_output['graph_db_results'])])
                        content = f"节点名称为{node_name}，图检索到的文档为{graphdoc}"
                    elif node_name == "generate_answer" or node_name == "direct_answer":
                        content = f"节点名称为{node_name}，回答完毕"
                        
                        # 提取来源信息
                        sources = node_output.get('answer_sources', [])
                        
                        # 存储messages类型的消息到数据库
                        extra_data = {
                            "node_name": node_name,
                            "sources": sources  # 将来源信息也存储到数据库
                        }
                        
                        latest_message = node_output['messages'][-1]  # 获取最新的一条消息
                        message_content = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
                        save_chat_message(
                            conversation_id=session_id,
                            role="assistant",
                            message_type="messages",
                            content=message_content,
                            extra_data=extra_data
                        )
                        
                        # 发送来源信息到前端
                        if sources:
                            yield {
                                "type": "sources",
                                "session_id": session_id,
                                "sources": sources
                            }
                    else:
                        content = f"节点名称为{node_name}"
                    
                    # 统一yield
                    yield {
                        "type": "node_update",
                        "session_id": session_id,
                        "node_name": node_name,
                        "content": content
                    }
                    
                    # 存储updates类型的消息到数据库（不检查长度）
                    extra_data = {"node_name": node_name}
                    save_chat_message(
                        conversation_id=session_id,
                        role="system",
                        message_type="updates",
                        content=content,
                        extra_data=extra_data
                    )
                    
                if mode =="messages":
                    chunkmessage,metadata=chunk
                    if chunkmessage.response_metadata and chunkmessage.response_metadata["finish_reason"] == "stop":
                        yield {
                        "type": "token",
                        "session_id": session_id,
                        "content": "\n"
                    }
                    if chunkmessage.content:
                        logger.info(f"（流式输出）消息: {chunkmessage.content}")
                        yield {
                            "type": "token",
                            "session_id": session_id,
                            "content": chunkmessage.content
                        }

            # 流式输出完成后,发送结束节点通知
            end_content = "节点名称为end,对话流程结束"
            yield {
                "type": "node_update",
                "session_id": session_id,
                "node_name": "end",
                "content": end_content
            }

            # 存储结束节点消息到数据库
            extra_data = {"node_name": "end"}
            save_chat_message(
                conversation_id=session_id,
                role="system",
                message_type="updates",
                content=end_content,
                extra_data=extra_data
            )



        except Exception as stream_error:
            logger.warning(f"流式输出失败，回退到普通模式: {str(stream_error)}")
            logger.exception("详细错误信息:")
            # 回退到普通调用
            try:
                result = rag_graph.invoke(input_data, context)
                if isinstance(result, dict) and "final_answer" in result and result["final_answer"]:
                    yield {
                        "type": "answer",
                        "session_id": session_id,
                        "content": result["final_answer"],
                        "sources": result.get("answer_sources", [])
                    }
                else:
                    # 如果没有final_answer，尝试从result中提取内容
                    content = str(result) if result else "处理完成，但未获得有效响应"
                    yield {
                        "type": "message",
                        "session_id": session_id,
                        "content": content
                    }
            except Exception as invoke_error:
                logger.error(f"普通调用也失败: {str(invoke_error)}")
                yield {
                    "type": "error",
                    "session_id": session_id,
                    "error": str(invoke_error),
                    "message": "处理失败"
                }
        
        # 发送完成信号
        yield {
            "type": "complete",
            "session_id": session_id,
            "message": "聊天处理完成"
        }
        
        logger.info("流式聊天处理完成")
        
    except Exception as e:
        logger.error(f"流式聊天处理失败: {str(e)}")
        logger.exception("详细错误信息:")
        
        yield {
            "type": "error",
            "error": str(e),
            "message": "流式聊天处理失败"
        }


async def get_chat_history_list(user_id: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    获取聊天历史列表
    
    Args:
        user_id: 用户ID
        conversation_id: 会话ID（可选，如果提供则获取指定会话的历史）
        
    Returns:
        Dict[str, Any]: 聊天历史数据
    """
    try:
        logger.info(f"获取用户 {user_id} 的聊天历史")
        
        # 验证用户ID
        if not user_id or not str(user_id).strip():
            return {
                "success": False,
                "error": "用户ID不能为空",
                "message": "获取聊天历史失败"
            }
        
        if conversation_id and str(conversation_id).strip():
            # 获取指定会话的历史记录
            logger.info(f"获取会话 {conversation_id} 的历史记录")
            
            # 验证对话是否存在
            conv_result = await conversation_service.get_conversation_by_id(conversation_id)
            if not conv_result.get("success"):
                return {
                    "success": False,
                    "error": "对话不存在",
                    "message": "指定的对话不存在"
                }
            
            try:
                # 直接从数据库获取聊天历史（不需要 RAGGraph）
                from backend.service.chat_history import get_chat_messages
                history_records = get_chat_messages(conversation_id)
                
                # 转换为前端需要的格式
                history = []
                for record in history_records:
                    history_item = {
                        'id': record['id'],
                        'conversation_id': record['conversation_id'],
                        'role': record['role'],
                        'type': record['type'],
                        'content': record['content']
                    }
                    
                    # 如果有额外数据，添加到历史项中
                    if record.get('extra_data'):
                        # 如果extra_data中有node_name，提取出来
                        if isinstance(record['extra_data'], dict) and 'node_name' in record['extra_data']:
                            history_item['node_name'] = record['extra_data']['node_name']
                    
                    history.append(history_item)
                
                logger.info(f"成功获取会话 {conversation_id} 的 {len(history)} 条历史记录")
                
                return {
                    "success": True,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "history": history,
                    "message": f"成功获取 {len(history)} 条聊天历史"
                }
            except Exception as db_error:
                logger.error(f"从数据库获取历史失败: {str(db_error)}")
                return {
                    "success": False,
                    "error": f"获取会话历史失败: {str(db_error)}",
                    "message": "获取聊天历史失败"
                }
        else:
            # 获取用户所有对话列表
            logger.info("获取用户所有对话列表")
            
            result = await conversation_service.get_conversations_by_user(
                user_id=user_id
            )
            
            if not result["success"]:
                logger.error(f"获取用户对话列表失败: {result.get('error', '未知错误')}")
                return {
                    "success": False,
                    "error": result.get('error', '未知错误'),
                    "message": "获取用户对话列表失败"
                }
            
            conversations = result["data"]["conversations"]
            conversation_list = []
            for conv in conversations:
                conversation_list.append({
                    "conversation_id": conv["conversation_id"],
                    "title": conv["title"],
                    "created_at": conv["created_at"],
                    "updated_at": conv["updated_at"]
                })
            
            logger.info(f"成功获取用户 {user_id} 的 {len(conversation_list)} 个对话")
            
            return {
                "success": True,
                "user_id": user_id,
                "conversations": conversation_list,
                "message": f"成功获取 {len(conversation_list)} 个对话"
            }
        
    except Exception as e:
        logger.error(f"获取聊天历史失败: {str(e)}")
        logger.exception("详细错误信息:")
        return {
            "success": False,
            "error": str(e),
            "message": "获取聊天历史失败"
        }


async def add_chat_history_list(user_id: str, conversation_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
    """
    添加聊天历史记录（直接保存到数据库，不使用 RAGGraph）
    
    Args:
        user_id: 用户ID
        conversation_id: 会话ID
        message: 消息内容，格式: {"role": "user/assistant", "content": "消息内容"}
        
    Returns:
        Dict[str, Any]: 操作结果
    """
    try:
        logger.info(f"为用户 {user_id} 添加聊天历史记录到会话 {conversation_id}")
        
        # 验证参数
        if not user_id or not str(user_id).strip():
            return {
                "success": False,
                "error": "用户ID不能为空",
                "message": "添加聊天历史失败"
            }
        
        if not conversation_id or not str(conversation_id).strip():
            return {
                "success": False,
                "error": "会话ID不能为空",
                "message": "添加聊天历史失败"
            }
        
        # 验证对话是否存在
        conv_result = await conversation_service.get_conversation_by_id(conversation_id)
        if not conv_result.get("success"):
            return {
                "success": False,
                "error": "对话不存在",
                "message": "指定的对话不存在"
            }
        
        # 验证消息格式
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            logger.error(f"消息格式无效: {message}")
            return {
                "success": False,
                "error": "消息格式无效，需要包含role和content字段",
                "message": "添加聊天历史失败"
            }
        
        # 提取消息信息
        role = message.get("role", "").lower()
        content = str(message.get("content", "")).strip()
        
        # 验证角色
        if role not in ["user", "assistant", "system"]:
            logger.error(f"不支持的消息角色: {role}")
            return {
                "success": False,
                "error": f"不支持的消息角色: {role}，仅支持user、assistant和system",
                "message": "添加聊天历史失败"
            }
        
        # 直接保存到数据库
        try:
            save_chat_message(
                conversation_id=conversation_id,
                role=role,
                message_type="messages",
                content=content,
                extra_data={"added_manually": True}
            )
            
            logger.info(f"成功添加消息到会话 {conversation_id}: {content[:50]}...")
            
            return {
                "success": True,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "message": "聊天历史添加成功",
                "added_message": {
                    "role": role,
                    "content": content
                }
            }
        except Exception as db_error:
            logger.error(f"保存消息到数据库失败: {str(db_error)}")
            return {
                "success": False,
                "error": f"保存消息失败: {str(db_error)}",
                "message": "添加聊天历史失败"
            }
        
    except Exception as e:
        logger.error(f"添加聊天历史失败: {str(e)}")
        logger.exception("详细错误信息:")
        return {
            "success": False,
            "error": str(e),
            "message": "添加聊天历史失败"
        }


async def create_conversation(user_id: str, title: str = None) -> Dict[str, Any]:
    """
    创建新对话
    
    Args:
        user_id: 用户ID
        title: 对话标题（可选）
        
    Returns:
        Dict[str, Any]: 创建结果
    """
    return await conversation_service.create_conversation(user_id, title)


async def update_conversation_title(conversation_id: str, title: str) -> Dict[str, Any]:
    """
    更新对话标题
    
    Args:
        conversation_id: 对话ID
        title: 新标题
        
    Returns:
        Dict[str, Any]: 更新结果
    """
    return await conversation_service.update_conversation_title(conversation_id, title)


async def delete_conversation(conversation_id: str) -> Dict[str, Any]:
    """
    删除对话
    
    Args:
        conversation_id: 对话ID
        
    Returns:
        Dict[str, Any]: 删除结果
    """
    return await conversation_service.delete_conversation(conversation_id)