from venv import logger
from langgraph.runtime import Runtime
from langgraph.prebuilt import create_react_agent
from ..states.raggraph_state import RAGGraphState
from ..contexts.raggraph_context import RAGContext
from ..models.raggraph_models import RetrievalMode, RetrievedDocument
from ..prompts.raggraph_prompt import (
    RAGGraphPrompts,
    RetrievalNeedDecision,
    SubquestionExpansion,
    RetrievalTypeDecision
)
from langmem import create_manage_memory_tool, create_search_memory_tool
from langchain_core.messages import AIMessage, ToolMessage
from ...config.log import get_logger
from backend.agent.tools.exceptions import (
    ToolCallException,
    ToolExecutionError,
    ToolNotFoundError,
    ToolExecutionTimeoutError
)
from backend.agent.tools.audit import get_audit_logger
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

class RAGNodes:
    """RAG图节点实现类

    包含所有RAG图的节点实现和路由逻辑
    """

    def __init__(self, llm=None, embedding_model=None, milvus_storage=None, memory_store=None, checkpointer=None, lightrag_storage=None, tools=None):
        """初始化RAG节点

        Args:
            llm: 语言模型
            embedding_model: 嵌入模型
            milvus_storage: Milvus存储实例
            memory_store: 记忆存储实例
            checkpointer: 检查点存储实例
            lightrag_storage: LightRAG存储实例
            tools: 可用工具列表（如风险评估工具）
        """
        self.llm = llm
        self.embedding_model = embedding_model
        self.milvus_storage = milvus_storage
        self.memory_store = memory_store
        self.checkpointer = checkpointer
        self.lightrag_storage = lightrag_storage
        self.tools = tools or []  # 工具列表
        self.logger = get_logger(__name__)
        self.tool_timeout = 30.0  # 工具执行超时（30秒）
        self.executor = ThreadPoolExecutor(max_workers=5)  # 线程池用于超时控制
    
    def _execute_tool_with_timeout(self, tool, tool_args, timeout=None):
        """带超时控制的工具执行
        
        Args:
            tool: 工具实例
            tool_args: 工具参数
            timeout: 超时时间（秒），默认使用 self.tool_timeout
        
        Returns:
            工具执行结果
        
        Raises:
            ToolExecutionTimeoutError: 工具执行超时
        """
        timeout = timeout or self.tool_timeout
        
        try:
            # 使用线程池执行工具，并设置超时
            future = self.executor.submit(tool.invoke, tool_args)
            result = future.result(timeout=timeout)
            return result
        except FutureTimeoutError:
            # 超时异常
            raise ToolExecutionTimeoutError(
                tool_name=tool.name,
                timeout=timeout,
                details=f"工具 '{tool.name}' 执行超过 {timeout} 秒"
            )

    # ==================== 节点实现 ====================

    def start_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        """开始节点

        RAG处理流程的入口节点，主要用于日志记录和状态确认。
        状态已在create_initial_rag_state函数中完成初始化。

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态
        """
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: START - 开始处理")

        return state

    def check_tool_needed_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        """判断是否需要工具调用节点

        使用LLM智能判断用户问题是否需要调用工具（如风险评估）。
        只有当用户提供了足够的参数时才会调用工具。

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态
        """
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: CHECK_TOOL_NEEDED - 判断是否需要工具")

        # 如果没有工具，直接跳过
        if not self.tools:
            self.logger.info("无可用工具，跳过工具检查")
            state["need_tool"] = False
            return state

        # 获取用户问题
        messages = state.get("messages", [])
        if not messages:
            state["need_tool"] = False
            return state

        user_question = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
        self.logger.info(f"分析问题: {user_question}")

        try:
            # 构建工具描述
            tool_descriptions = []
            for tool in self.tools:
                tool_descriptions.append(f"- {tool.name}: {tool.description}")
            tools_text = "\n".join(tool_descriptions)

            # 使用LLM判断是否需要工具
            prompt = f"""你是一个智能助手，需要判断用户问题是否需要调用工具。

可用工具：
{tools_text}

用户问题：{user_question}

判断规则：
1. 用户是否明确请求进行风险评估、计算等需要工具的任务
2. 对于糖尿病风险评估：
   - **必需参数**：年龄(age)、BMI
   - **可选参数**：腰围、血压、家族史、体育活动、吸烟、饮食质量（这些有默认值）
3. 对于高血压风险评估：
   - **必需参数**：年龄、收缩压、舒张压
   - **可选参数**：家族史、吸烟、盐分摄入（这些有默认值）
4. 只要用户提供了必需参数，就应该调用工具（可选参数会使用默认值）

只有同时满足以下条件才返回需要工具：
- 用户明确需要工具功能（如"风险评估"）
- 提供了该工具的**必需参数**

请用JSON格式回答：
{{
  "need_tool": true/false,
  "tool_name": "工具名称或null",
  "reason": "判断理由",
  "missing_params": ["缺少的必需参数列表，如果有"]
}}"""

            response = self.llm.invoke(prompt)
            response_text = response.content

            # 解析JSON响应
            import re
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
                need_tool = decision.get("need_tool", False)
                tool_name = decision.get("tool_name")
                reason = decision.get("reason", "")

                state["need_tool"] = need_tool
                state["selected_tool"] = tool_name
                self.logger.info(f"LLM判断: need_tool={need_tool}, tool={tool_name}, reason={reason}")
            else:
                # 解析失败，默认不使用工具
                state["need_tool"] = False
                self.logger.warning("无法解析LLM响应，默认不使用工具")

        except Exception as e:
            self.logger.error(f"工具判断失败: {e}")
            state["need_tool"] = False

        return state

    def tool_calling_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        """工具调用节点

        执行工具调用，获取结果并格式化为友好的回答。

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态
        """
        self.logger.info("="*50)
        self.logger.info("[RAG Graph] 节点: TOOL_CALLING - 执行工具调用")

        messages = state.get("messages", [])
        user_question = messages[-1].content if messages and hasattr(messages[-1], 'content') else ""
        selected_tool = state.get("selected_tool")

        self.logger.info(f"用户问题: {user_question}")
        self.logger.info(f"选中的工具: {selected_tool}")

        try:
            # 绑定工具到LLM
            llm_with_tools = self.llm.bind_tools(self.tools)

            # 构建更明确的提示词，强制LLM调用工具
            tool_names = [tool.name for tool in self.tools]
            prompt = f"""用户请求进行风险评估。

用户问题：{user_question}

你必须使用以下工具之一来回答：{tool_names}

请直接调用工具，从用户问题中提取参数。

参数提取指导：

对于 diabetes_risk_assessment（糖尿病风险评估）：
- age：提取年龄数字
- bmi：提取BMI数值（可能是小数）
- blood_pressure：如果提到“血压偏高”或“高血压”，设置 blood_pressure="偏高"
- family_history：如果提到“有家族史”或类似表述，设置 family_history=True

对于 hypertension_risk_assessment（高血压风险评估）：
- age：提取年龄数字
- systolic_bp：提取收缩压（高压）数字，单位mmHg
- diastolic_bp：提取舒张压（低压）数字，单位mmHg
  例如：“血压140/90” → systolic_bp=140, diastolic_bp=90
- family_history：如果提到高血压家族史，设置 family_history=True
- smoking：如果提到吸烟，设置 smoking=True

必须调用工具，不能只是返回文本回答！"""

            # 调用LLM（带工具）
            self.logger.info("调用LLM，期待工具调用...")
            response = llm_with_tools.invoke(prompt)

            # 检查是否有工具调用
            if hasattr(response, 'tool_calls') and response.tool_calls:
                self.logger.info(f"LLM请求调用 {len(response.tool_calls)} 个工具")

                # 执行工具调用
                tool_results = []
                audit_logger = get_audit_logger()
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    self.logger.info(f"调用工具: {tool_name}, 参数: {tool_args}")
                    
                    # 记录开始时间
                    start_time = time.time()
                    tool_result = None
                    tool_error = None

                    # 查找并执行工具
                    tool_found = False
                    for tool in self.tools:
                        if tool.name == tool_name:
                            tool_found = True
                            try:
                                # 使用带超时的执行方法
                                tool_result = self._execute_tool_with_timeout(tool, tool_args)
                                tool_results.append(tool_result)
                                self.logger.info(f"工具执行成功: {tool_name}, 结果: {tool_result}")
                            except ToolExecutionTimeoutError as timeout_error:
                                # 超时异常
                                self.logger.error(f"工具执行超时: {timeout_error.to_dict()}")
                                tool_error = timeout_error.to_dict()
                                
                                state["error"] = timeout_error.to_dict()
                                state["final_answer"] = timeout_error.message
                                state["messages"] = [AIMessage(content=timeout_error.message)]
                            except Exception as tool_exec_error:
                                # 其他执行错误
                                error = ToolExecutionError(
                                    tool_name=tool_name,
                                    error_message=str(tool_exec_error),
                                    details=f"{type(tool_exec_error).__name__}: {tool_exec_error}"
                                )
                                self.logger.error(f"工具执行失败: {error.to_dict()}")
                                tool_error = error.to_dict()
                                
                                state["error"] = error.to_dict()
                                state["final_answer"] = error.message
                                state["messages"] = [AIMessage(content=error.message)]
                            finally:
                                # 计算执行时间
                                execution_time_ms = (time.time() - start_time) * 1000
                                
                                # 审计日志
                                audit_logger.log_tool_call(
                                    conversation_id=state.get("session_id", "unknown"),
                                    user_id=state.get("user_id", "unknown"),
                                    tool_name=tool_name,
                                    tool_args=tool_args,
                                    result=tool_result,
                                    error=tool_error,
                                    execution_time_ms=execution_time_ms
                                )
                                
                                # 如果有错误，直接返回
                                if tool_error:
                                    return state
                            break
                    
                    # 工具未找到
                    if not tool_found:
                        available_tools = [t.name for t in self.tools]
                        error = ToolNotFoundError(
                            tool_name=tool_name,
                            available_tools=available_tools
                        )
                        self.logger.error(f"工具未找到: {error.to_dict()}")
                        
                        # 审计日志
                        execution_time_ms = (time.time() - start_time) * 1000
                        audit_logger.log_tool_call(
                            conversation_id=state.get("session_id", "unknown"),
                            user_id=state.get("user_id", "unknown"),
                            tool_name=tool_name,
                            tool_args=tool_args,
                            error=error.to_dict(),
                            execution_time_ms=execution_time_ms
                        )
                        
                        state["error"] = error.to_dict()
                        state["final_answer"] = error.message
                        state["messages"] = [AIMessage(content=error.message)]
                        return state

                if tool_results:
                    # 使用LLM将工具结果转化为友好的回答
                    final_prompt = f"""基于以下工具执行结果，请给用户一个清晰、专业、友好的回答。

用户问题：{user_question}

工具结果：
{json.dumps(tool_results, ensure_ascii=False, indent=2)}

请：
1. 用通俗易懂的语言解释结果
2. 突出显示关键信息（风险等级、评分等）
3. 给出具体的健康建议
4. 使用Markdown格式，结构清晰"""

                    # 使用invoke()而不是stream()
                    # 原因：LangGraph的astream会自动处理流式输出
                    # 在节点内部使用stream()会导致所有chunk都带langgraph_node metadata
                    final_response = self.llm.invoke(final_prompt)

                    # 构建完整的来源信息（支持多工具）
                    sources = []
                    for i, (tool_call, result) in enumerate(zip(response.tool_calls, tool_results)):
                        tool_name = tool_call['name']
                        sources.append({
                            "index": i + 1,
                            "tool_name": tool_name,
                            "document_name": f"工具: {tool_name}",
                            "content": f"风险等级: {result.get('risk_level', 'N/A')}, 评分: {result.get('risk_score', 'N/A')}",
                            "retrieval_mode": "tool",
                            "metadata": {
                                "tool_args": tool_call['args'],
                                "risk_level": result.get('risk_level'),
                                "risk_score": result.get('risk_score'),
                                "risk_factors": result.get('risk_factors', []),
                                "recommendations": result.get('recommendations', [])
                            }
                        })

                    state["final_answer"] = final_response.content
                    state["answer_sources"] = sources
                    state["messages"] = [AIMessage(content=final_response.content)]
                    self.logger.info("✅ 工具调用成功，生成回答")
                    return state
            
            # 如果LLM没有调用工具，记录详细信息
            self.logger.warning("⚠️ LLM没有调用工具！")
            self.logger.warning(f"LLM响应: {response}")
            self.logger.warning(f"LLM响应类型: {type(response)}")
            self.logger.warning(f"hasattr tool_calls: {hasattr(response, 'tool_calls')}")
            if hasattr(response, 'tool_calls'):
                self.logger.warning(f"tool_calls 内容: {response.tool_calls}")
            
            # 没有工具调用，返回提示
            state["final_answer"] = "抱歉，我无法从您的问题中提取足够的参数来进行评估。请提供更详细的信息。"
            state["messages"] = [AIMessage(content=state["final_answer"])]

        except ToolCallException as e:
            # 已经是结构化异常，直接使用
            self.logger.error(f"工具调用异常: {e.to_dict()}")
            state["error"] = e.to_dict()
            state["final_answer"] = e.message
            state["messages"] = [AIMessage(content=e.message)]
        except Exception as e:
            # 未预期的异常，包装为通用错误
            import traceback
            self.logger.error(f"工具调用失败: {e}")
            self.logger.error(traceback.format_exc())
            
            from backend.agent.tools.exceptions import ToolCallErrorCode
            error = ToolCallException(
                code=ToolCallErrorCode.UNKNOWN_ERROR,
                message="抱歉，工具调用时遇到未知问题",
                details=traceback.format_exc(),
                metadata={"original_error": str(e)}
            )
            state["error"] = error.to_dict()
            state["final_answer"] = error.message
            state["messages"] = [AIMessage(content=error.message)]

        return state

    def check_retrieval_needed_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        """判断是否需要检索节点

        根据retrieval_mode和LLM判断是否需要进行检索：
        1. 如果retrieval_mode为NO_RETRIEVAL，直接设置为False
        2. 否则调用LLM进行智能判断

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态
        """
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: CHECK_RETRIEVAL_NEEDED - 判断是否需要检索")
        
        retrieval_mode = state['retrieval_mode']


        # 检查retrieval_mode是否为NO_RETRIEVAL
        if retrieval_mode == RetrievalMode.NO_RETRIEVAL:
            self.logger.info("跳过检索")
            state["need_retrieval"] = False
            state["need_retrieval_reason"] = "用户设置为不需要检索模式"
            # 设置原始问题
            messages = state.get("messages", [])
            if messages:
                latest_message = messages[-1].content 
                state["original_question"] = latest_message
            return state

        # 如果不是AUTO模式，直接设置需要检索
        if retrieval_mode != RetrievalMode.AUTO:
            self.logger.info("直接设置需要检索")
            state["need_retrieval"] = True
            state["need_retrieval_reason"] = f"用户设置为{retrieval_mode}检索模式，直接进行检索"
            # 设置原始问题
            messages = state.get("messages", [])
            if messages:
                latest_message = messages[-1].content 
                state["original_question"] = latest_message
            return state

        # AUTO模式：调用LLM进行检索需求判断
        self.logger.info("AUTO模式，调用LLM判断...")
        try:
            # 获取最新消息
            messages = state.get("messages", [])
            latest_message = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

            # 构建提示词
            prompt_template = RAGGraphPrompts.get_retrieval_need_judgment_prompt()
            prompt = prompt_template.format(question=latest_message)

            # 使用结构化输出调用LLM
            try:
                structured_llm = self.llm.with_structured_output(RetrievalNeedDecision)
                decision = structured_llm.invoke(prompt)

                state["need_retrieval"] = decision.need_retrieval
                self.logger.info(f"LLM判断结果: {decision.need_retrieval}")
                state["need_retrieval_reason"] = decision.reasoning

                # 更新提取的核心问题
                if decision.extracted_question and decision.extracted_question.strip():
                    state["original_question"] = decision.extracted_question.strip()
                else:
                    state["original_question"] = latest_message

            except Exception as parse_error:
                self.logger.error(f"结构化输出调用失败: {parse_error}")
                # 解析失败时默认需要检索
                state["need_retrieval"] = True
                state["need_retrieval_reason"] = f"LLM结构化输出调用失败: {str(parse_error)}"
                state["original_question"] = latest_message

        except Exception as e:
            self.logger.error(f"检索需求判断失败: {e}")
            # 出错时默认需要检索
            state["need_retrieval"] = True
            state["need_retrieval_reason"] = f"检索需求判断过程出错: {str(e)}"

        return state

    def expand_subquestions_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        """由原始问题扩展子问题节点

        根据原始问题，调用LLM将其分解为多个具体的子问题，
        以便进行更精确的信息检索。

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态，包含扩展的子问题列表
        """
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: EXPAND_SUBQUESTIONS - 扩展子问题")

        # 获取原始问题
        original_question = state.get("original_question", "")
        if not original_question:
            self.logger.warning("原始问题为空，无法扩展子问题")
            state["subquestions"] = []
            return state

        self.logger.info(f"原始问题: {original_question}")

        try:
            # 获取子问题扩展提示词
            prompt_template = RAGGraphPrompts.get_subquestion_expansion_prompt()
            prompt = prompt_template.format(question=original_question)

            # 使用结构化输出调用LLM
            structured_llm = self.llm.with_structured_output(SubquestionExpansion)
            expansion_result = structured_llm.invoke(prompt)

            # 获取子问题列表
            subquestions = expansion_result.subquestions

            # 验证和清理子问题
            if subquestions:
                # 过滤空字符串和重复项
                cleaned_subquestions = []
                for sq in subquestions:
                    if sq and sq.strip() and sq.strip() not in cleaned_subquestions:
                        cleaned_subquestions.append(sq.strip())

                state["subquestions"] = cleaned_subquestions
                self.logger.info(f"成功扩展 {len(cleaned_subquestions)} 个子问题")
            else:
                # 如果没有生成子问题，使用原问题
                state["subquestions"] = [original_question]
                self.logger.info("未生成子问题，使用原问题")

        except Exception as e:
            self.logger.error(f"子问题扩展失败: {e}")
            # 失败时使用原问题作为唯一子问题
            state["subquestions"] = [original_question]

        return state

    def classify_question_type_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        """判断检索类型节点

        根据context中的retrieval_mode配置决定使用哪种检索方式。
        如果是AUTO模式，调用LLM进行智能判断并更新retrieval_mode。

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态
        """
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: CLASSIFY_QUESTION_TYPE - 判断检索类型")


        # 如果是AUTO模式，调用LLM进行智能判断
        if state["retrieval_mode"] == RetrievalMode.AUTO:
            self.logger.info("AUTO模式，开始智能判断检索类型...")

            try:
                # 获取原始问题
                original_question = state.get("original_question", "")
                if not original_question:
                    self.logger.warning("未找到原始问题，默认使用向量检索")
                    state["retrieval_mode"] = RetrievalMode.VECTOR_ONLY
                    return state

                # 构建提示词
                prompt_template = RAGGraphPrompts.get_retrieval_type_judgment_prompt()
                prompt = prompt_template.format(question=original_question)

                # 调用LLM进行判断
                if not self.llm:
                    self.logger.warning("未找到LLM实例，默认使用向量检索")
                    state["retrieval_mode"] = RetrievalMode.VECTOR_ONLY
                    return state

                # 使用结构化输出调用LLM
                try:
                    structured_llm = self.llm.with_structured_output(RetrievalTypeDecision)
                    decision = structured_llm.invoke(prompt)
                    state["retrieval_mode_reason"] = decision.reasoning
                    # 根据LLM判断结果更新检索模式
                    if decision.retrieval_type == "vector_only":
                        state["retrieval_mode"] = RetrievalMode.VECTOR_ONLY
                        self.logger.info(f"LLM判断：使用向量检索 - {decision.reasoning}")
                    elif decision.retrieval_type == "graph_only":
                        state["retrieval_mode"] = RetrievalMode.GRAPH_ONLY
                        self.logger.info(f"LLM判断：使用图检索 - {decision.reasoning}")
                    else:
                        self.logger.warning(f"未知的检索类型 {decision.retrieval_type}，默认使用向量检索")
                        state["retrieval_mode"] = RetrievalMode.VECTOR_ONLY

                except Exception as parse_error:
                    self.logger.error(f"结构化输出调用失败: {parse_error}")
                    # 解析失败时默认使用向量检索
                    state["retrieval_mode"] = RetrievalMode.VECTOR_ONLY

            except Exception as e:
                self.logger.error(f"检索类型判断失败: {e}")
                # 出错时默认使用向量检索
                state["retrieval_mode"] = RetrievalMode.VECTOR_ONLY
        else:
            self.logger.info(f"未进行智能判断，保持当前检索模式: {state['retrieval_mode']}")
            state["retrieval_mode_reason"] = f"未进行智能判断,保持当前检索模式{state['retrieval_mode']}"

        self.logger.info(f"最终检索模式: {state['retrieval_mode']}")
        return state

    def vector_db_retrieval_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        """向量数据库检索节点

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态
        """
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: VECTOR_DB_RETRIEVAL - 向量数据库检索")

        # 检查MilvusStorage是否可用
        if not self.milvus_storage:
            self.logger.warning("MilvusStorage未初始化，跳过向量检索")
            state["retrieved_docs"] = []
            state["vector_db_results"] = []
            return state

        # 获取检索查询
        original_question = state.get("original_question", "")
        if not original_question:
            self.logger.warning("未找到原始问题，跳过向量检索")
            state["retrieved_docs"] = []
            state["vector_db_results"] = []
            return state

        try:
            # 从context获取检索配置
            context = runtime.context
            max_docs = context.max_retrieval_docs if context else 3

            # 创建混合检索器
            hybrid_retriever = self.milvus_storage.create_hybrid_retriever(
                search_kwargs={"k": max_docs}
            )

            # 收集所有需要检索的问题
            questions_to_search = [original_question]

            # 添加子问题到检索列表
            subquestions = state.get("subquestions", [])
            if subquestions:
                questions_to_search.extend(subquestions)
                self.logger.info(f"将对 {len(questions_to_search)} 个问题进行检索（1个原始问题 + {len(subquestions)}个子问题）")
            else:
                self.logger.info("将对原始问题进行检索")

            # 依次对每个问题进行检索
            all_retrieved_docs = []
            for i, question in enumerate(questions_to_search):
                try:
                    docs = hybrid_retriever.invoke(question)
                    all_retrieved_docs.extend(docs)
                except Exception as search_error:
                    self.logger.error(f"问题 {i+1} 检索失败: {search_error}")

            self.logger.info(f"总共检索到 {len(all_retrieved_docs)} 个文档")

            # 转换为RetrievedDocument格式
            converted_docs = []
            for doc in all_retrieved_docs:
                retrieved_doc = RetrievedDocument(
                    page_content=doc.page_content,
                    metadata=doc.metadata
                )
                converted_docs.append(retrieved_doc)
                # 调试日志：查看每个文档的metadata
                self.logger.info(f"检索文档 metadata: document_name={doc.metadata.get('document_name')}, "
                               f"chunk_index={doc.metadata.get('chunk_index')}, "
                               f"content_length={len(doc.page_content)}, "
                               f"pk={doc.metadata.get('pk')}")

            # 根据pk值进行去重处理
            unique_docs = []
            seen_pks = set()

            for doc in converted_docs:
                # 从metadata中获取pk值
                pk = doc.metadata.get("pk") or doc.metadata.get("id")

                if pk is not None:
                    # 如果有pk值，检查是否已见过
                    if pk not in seen_pks:
                        seen_pks.add(pk)
                        unique_docs.append(doc)
                else:
                    # 如果没有pk值，保留文档（但记录警告）
                    unique_docs.append(doc)

            self.logger.info(f"去重后文档数: {len(unique_docs)}")

            # 更新状态
            state["retrieved_docs"] = unique_docs
            state["vector_db_results"] = unique_docs

        except Exception as e:
            self.logger.error(f"向量检索失败: {e}")
            # 检索失败时设置空结果
            state["retrieved_docs"] = []
            state["vector_db_results"] = []

        return state

    async def graph_db_retrieval_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        """图数据库检索节点

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态
        """
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: GRAPH_DB_RETRIEVAL - 图数据库检索")

        # 获取查询相关信息
        original_question = state.get("original_question", "")
        subquestions = state.get("subquestions", [])
        query_text = original_question if original_question else (subquestions[0] if subquestions else "")

        self.logger.info(f"执行图数据库检索，查询: {query_text}")

        try:

            # 执行图数据库检索 - 使用global模式进行图检索
            result = await self.lightrag_storage.query(
                query=query_text,
                mode="hybrid",  # global模式专门用于图数据库检索
                only_need_prompt=True
            )

            # self.logger.info(f"图数据库检索原始结果: {result}")

            # 处理检索结果,只保留Document Chunks部分
            if result and len(result.strip()) > 0:
                # 提取Document Chunks部分
                dc_marker = "-----Document Chunks(DC)-----"
                if dc_marker in result:
                    # 只保留DC标记之后的内容
                    result = result.split(dc_marker, 1)[1].strip()
                    #self.logger.info(f"提取Document Chunks后的结果: {result}")

                # 将检索结果转换为文档格式
                graph_doc = RetrievedDocument(
                    page_content=result,
                    metadata={
                        "source": "lightrag_graph",
                        "retrieval_mode": "global",
                        "document_name": "知识图谱检索结果"
                    }
                )

                state["retrieved_docs"] = [graph_doc]
                state["graph_db_results"] = [graph_doc]

                self.logger.info(f"图数据库检索成功，结果长度: {len(result)}")
            else:
                self.logger.warning("图数据库检索未返回结果")
                state["retrieved_docs"] = []
                state["graph_db_results"] = []

        except Exception as e:
            self.logger.error(f"图数据库检索失败: {e}")
            # 检索失败时设置空结果
            state["retrieved_docs"] = []
            state["graph_db_results"] = []

        return state



    def generate_answer_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        """生成答案节点

        基于检索到的文档和用户问题，调用LLM生成最终答案。

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态，包含生成的答案和更新的messages
        """
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: GENERATE_ANSWER - 生成答案")

        # 获取原始问题
        original_question = state.get("original_question", "")
        retrieved_docs = state.get("retrieved_docs", [])
        
        self.logger.info(f"回答问题: {original_question}")
        self.logger.info(f"可用文档数量: {len(retrieved_docs)}")

        try:
            # 准备文档内容
            documents_text = ""
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs):
                    source = doc.metadata.get("document_name", f"文档{i+1}")
                    documents_text += f"\n[文档 {i+1} - {source}]:\n{doc.page_content}\n"
            else:
                documents_text = "暂无检索到的相关文档。"

            # 获取答案生成提示词
            prompt_template = RAGGraphPrompts.get_answer_generation_prompt()
            prompt = prompt_template.format(
                question=original_question,
                documents=documents_text,
                doc_count=len(retrieved_docs)
            )

            # 调用LLM生成答案
            answer_result = self.llm.invoke(prompt)
            answer_content = answer_result.content
            
            # 提取文档来源信息
            sources = []
            for i, doc in enumerate(retrieved_docs):
                retrieval_source = doc.metadata.get("source", "vector")
                
                if retrieval_source == "lightrag_graph":
                    content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                else:
                    content_preview = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                
                source_info = {
                    "index": i + 1,
                    "document_name": doc.metadata.get("document_name", f"文档{i+1}"),
                    "content": content_preview,
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "retrieval_mode": retrieval_source,
                    "content_length": len(doc.page_content)
                }
                sources.append(source_info)
            
            # 更新状态
            state["final_answer"] = answer_content
            state["answer_sources"] = sources
            state["messages"] = [answer_result]

            self.logger.info(f"答案生成成功，包含 {len(sources)} 个来源")

        except Exception as e:
            self.logger.error(f"答案生成失败: {e}")
            error_answer = "抱歉，在生成答案时遇到了技术问题。请稍后重试。"
            state["final_answer"] = error_answer
            state["answer_sources"] = []

        return state

    def direct_answer_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        """直接回答节点（支持短期对话记忆）

        对于不需要检索的常规问题，直接使用LLM生成答案，
        并结合当前会话内的对话历史（短期记忆）进行回答。

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态
        """
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: DIRECT_ANSWER - 直接回答")

        all_messages = state.get("messages", [])
        if not all_messages:
            self.logger.warning("没有可用的消息")
            state["final_answer"] = "抱歉，我没有收到您的问题。"
            return state

        latest_message = all_messages[-1]
        user_question = latest_message.content if hasattr(latest_message, "content") else str(latest_message)
        self.logger.info(f"用户问题: {user_question}")

        # 构建对话历史
        try:
            trimmed_messages = all_messages[-20:] if len(all_messages) > 20 else all_messages
            history_lines = []
            for msg in trimmed_messages:
                role = getattr(msg, "type", "user")
                content = msg.content if hasattr(msg, "content") else str(msg)
                if not content:
                    continue
                prefix = "用户" if role in ["human", "user"] else "助手" if role in ["ai", "assistant"] else "系统"
                history_lines.append(f"{prefix}: {content}")

            conversation_history = "\n".join(history_lines)
            if not conversation_history.strip():
                conversation_history = "（当前是本次对话的第一条消息）"

        except Exception as e:
            self.logger.warning(f"构建对话历史失败: {e}")
            conversation_history = "（无法获取对话历史）"

        # 生成回答
        try:
            prompt_template = RAGGraphPrompts.get_direct_answer_prompt()
            prompt = prompt_template.format(
                question=user_question,
                conversation_history=conversation_history
            )
            response = self.llm.invoke(prompt)
            answer = response.content

            state["final_answer"] = answer
            state["messages"] = [AIMessage(content=answer)]
            self.logger.info("直接回答生成成功")

        except Exception as e:
            self.logger.error(f"直接回答失败: {e}")
            error_answer = "抱歉，我在处理您的问题时遇到了问题。请稍后重试。"
            state["final_answer"] = error_answer
            state["messages"] = [AIMessage(content=error_answer)]

        return state

    # ====================记忆功能版本（已注释） ====================
    """
    def direct_answer_node_with_memory(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        \"\"\"直接回答节点，集成langmem记忆管理（已弃用）

        对于不需要检索的常规问题，使用create_react_agent模式，
        将记忆管理工具集成到agent中，让LLM自主决定何时使用记忆功能。
        适用于一般性问题、闲聊、简单计算等场景。

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态，包含生成的答案
        \"\"\"
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: DIRECT_ANSWER - 直接回答（集成记忆管理）")

        # 手动裁剪消息，只保留最新的20条
        all_messages = state.get("messages", [])
        if len(all_messages) > 20:
            trimmed_messages = all_messages[-20:]
            self.logger.info(f"消息裁剪: {len(all_messages)} -> {len(trimmed_messages)} 条")
        else:
            trimmed_messages = all_messages

        # 获取最新的用户消息
        latest_message = trimmed_messages[-1]

        # 获取记忆管理agent的提示词
        prompt = RAGGraphPrompts.get_direct_answer_memory_prompt()

        self.logger.info("创建带记忆功能的React Agent...")

        # 创建记忆管理工具 - 使用用户特定的命名空间
        # 获取用户ID，确保记忆隔离
        context = runtime.context
        user_id = context.user_id
        user_namespace = ("memories", user_id)

        tools = [
            # 记忆管理工具 - 可以创建、更新、删除记忆
            # 使用用户特定的命名空间实现数据隔离
            create_manage_memory_tool(namespace=user_namespace, store=self.memory_store),
            # 记忆搜索工具 - 可以搜索相关记忆
            create_search_memory_tool(namespace=user_namespace, store=self.memory_store),
        ]

        # 创建react agent
        agent = create_react_agent(
            self.llm,
            tools=tools,
            prompt=prompt,
            # checkpointer=self.checkpointer,
            # store=self.memory_store
        )

        # 准备agent输入 - 只传递最新的用户消息
        agent_input = {
            "messages": [latest_message]
        }

        # 获取配置信息（包含用户ID和会话ID）
        context = runtime.context
        config = context.get_langgraph_config()

        # 调用agent处理
        self.logger.info("调用React Agent处理用户问题...")
        agent_result = agent.invoke(agent_input, config=config)

        # 获取最终回答 - 使用最新的消息
        agent_messages = agent_result.get("messages", [])
        if agent_messages:
            # 直接使用最后一条消息作为最终答案
            final_answer = agent_messages[-1].content

            self.logger.info("React Agent回答生成成功")

            # 更新状态 - 只添加最新的agent消息
            latest_agent_message = agent_messages[-1]
            state["messages"] = [latest_agent_message]
            state["final_answer"] = final_answer

        return state
    """

    # ==================== 路由函数 ====================

    def route_retrieval_needed(self, state: RAGGraphState) -> str:
        """路由：是否需要检索

        Args:
            state: 当前状态

        Returns:
            路由目标
        """
        need_retrieval = state.get("need_retrieval", False)
        self.logger.info(f"[RAG Graph] 路由检查: need_retrieval = {need_retrieval}")

        if need_retrieval:
            self.logger.info("路由决策: 需要检索 -> check_tool_needed")
            return "need_retrieval"
        else:
            self.logger.info("路由决策: 无需检索 -> check_tool_needed")
            return "no_retrieval"

    def route_tool_needed(self, state: RAGGraphState) -> str:
        """路由：是否需要工具

        Args:
            state: 当前状态

        Returns:
            路由目标
        """
        need_tool = state.get("need_tool", False)
        self.logger.info(f"[RAG Graph] 路由检查: need_tool = {need_tool}")

        if need_tool:
            self.logger.info("路由决策: 需要工具 -> tool_calling")
            return "need_tool"
        else:
            self.logger.info("路由决策: 不需要工具 -> check_retrieval_needed")
            return "no_tool"

    def route_question_type(self, state: RAGGraphState) -> str:
        """路由：检索类型分类

        根据retrieval_mode决定使用哪种检索方式

        Args:
            state: 当前状态

        Returns:
            路由目标
        """
        retrieval_mode = state["retrieval_mode"]

        if retrieval_mode == RetrievalMode.VECTOR_ONLY:
            return "vector_db"
        elif retrieval_mode == RetrievalMode.GRAPH_ONLY:
            return "graph_db"
        elif retrieval_mode == RetrievalMode.AUTO:
            # AUTO模式默认使用向量检索，可以根据需要扩展智能判断逻辑
            return "vector_db"
        else:
            # 默认使用向量数据库
            return "vector_db"