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
    RetrievalTypeDecision,
    ToolSkillDecision
)
from langmem import create_manage_memory_tool, create_search_memory_tool
from langchain_core.messages import AIMessage, ToolMessage
from ...config.log import get_logger
from backend.agent.tools.exceptions import (
    ToolCallException,
    ToolExecutionError,
    ToolNotFoundError,
    ToolExecutionTimeoutError,
    ToolValidationError
)
from backend.agent.tools.audit import get_audit_logger
import asyncio
import json
import os
import time
import math
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from urllib.parse import urlparse, unquote
from backend.config.oss import get_presigned_url_for_download

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
        self.rrf_k = float(os.getenv("RAG_RRF_K", "60"))
        self.graph_source_weight = float(os.getenv("RAG_GRAPH_SOURCE_WEIGHT", "1.1"))
        self.vector_source_weight = float(os.getenv("RAG_VECTOR_SOURCE_WEIGHT", "1.0"))
        self.mmr_lambda = float(os.getenv("RAG_MMR_LAMBDA", "0.75"))
        self.enable_semantic_rerank = os.getenv("RAG_ENABLE_SEMANTIC_RERANK", "true").lower() in {"true", "1", "yes", "on"}
        self.assessment_skill_name = "health_risk_assessment"
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.assessment_profiles = {
            "hypertension_risk_assessment": {
                "required_params": ["age", "systolic_bp", "diastolic_bp"],
                "display_name": "高血压风险评估",
                "trigger_keywords": ["高血压", "血压", "收缩压", "舒张压", "高压", "低压"]
            },
            "diabetes_risk_assessment": {
                "required_params": ["age", "bmi"],
                "display_name": "糖尿病风险评估",
                "trigger_keywords": ["糖尿病", "血糖", "bmi", "体重指数"]
            }
        }
    
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

    def _extract_hypertension_slots(self, question: str) -> dict:
        slots = {}
        age_match = re.search(r"(?:年龄|age)\s*[:：]?\s*(\d{1,3})", question, re.IGNORECASE)
        if not age_match:
            age_match = re.search(r"(\d{1,3})\s*岁", question)
        if age_match:
            slots["age"] = int(age_match.group(1))
        bp_pair_match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", question)
        if bp_pair_match:
            slots["systolic_bp"] = int(bp_pair_match.group(1))
            slots["diastolic_bp"] = int(bp_pair_match.group(2))
        else:
            systolic_match = re.search(r"(?:收缩压|高压)\s*[:：]?\s*(\d{2,3})", question)
            diastolic_match = re.search(r"(?:舒张压|低压)\s*[:：]?\s*(\d{2,3})", question)
            if systolic_match:
                slots["systolic_bp"] = int(systolic_match.group(1))
            if diastolic_match:
                slots["diastolic_bp"] = int(diastolic_match.group(1))
        return slots

    def _extract_diabetes_slots(self, question: str) -> dict:
        slots = {}
        age_match = re.search(r"(?:年龄|age)\s*[:：]?\s*(\d{1,3})", question, re.IGNORECASE)
        if not age_match:
            age_match = re.search(r"(\d{1,3})\s*岁", question)
        if age_match:
            slots["age"] = int(age_match.group(1))
        bmi_match = re.search(r"(?:bmi|体重指数)\s*[:：]?\s*(\d{1,2}(?:\.\d{1,2})?)", question, re.IGNORECASE)
        if bmi_match:
            slots["bmi"] = float(bmi_match.group(1))
        return slots

    def _extract_required_slots(self, tool_name: str, question: str) -> dict:
        if tool_name == "hypertension_risk_assessment":
            return self._extract_hypertension_slots(question)
        if tool_name == "diabetes_risk_assessment":
            return self._extract_diabetes_slots(question)
        return {}

    def _missing_required_params(self, tool_name: str, question: str) -> list[str]:
        profile = self.assessment_profiles.get(tool_name, {})
        required_params = profile.get("required_params", [])
        if not required_params:
            return []
        extracted_slots = self._extract_required_slots(tool_name, question)
        return [param for param in required_params if param not in extracted_slots]

    def _is_assessment_skill_candidate(self, question: str) -> bool:
        question_text = (question or "").lower()
        generic_keywords = ["评估", "风险", "风险评估", "计算"]
        if any(keyword in question_text for keyword in generic_keywords):
            return True
        for profile in self.assessment_profiles.values():
            for keyword in profile.get("trigger_keywords", []):
                if keyword.lower() in question_text:
                    return True
        return False

    def _build_missing_params_answer(self, tool_name: str, missing_params: list[str]) -> str:
        display_name = self.assessment_profiles.get(tool_name, {}).get("display_name", tool_name)
        missing_text = "、".join(missing_params) if missing_params else "必需参数"
        return f"要进行{display_name}，还需要补充以下参数：{missing_text}。请补充后我将自动继续评估。"

    def _validate_tool_args(self, tool_name: str, tool_args: dict) -> dict:
        errors = {}
        if tool_name == "hypertension_risk_assessment":
            age = tool_args.get("age")
            systolic_bp = tool_args.get("systolic_bp")
            diastolic_bp = tool_args.get("diastolic_bp")
            if not isinstance(age, int) or age < 1 or age > 120:
                errors["age"] = "年龄需为1-120的整数"
            if not isinstance(systolic_bp, int) or systolic_bp < 60 or systolic_bp > 260:
                errors["systolic_bp"] = "收缩压需为60-260的整数"
            if not isinstance(diastolic_bp, int) or diastolic_bp < 30 or diastolic_bp > 180:
                errors["diastolic_bp"] = "舒张压需为30-180的整数"
        elif tool_name == "diabetes_risk_assessment":
            age = tool_args.get("age")
            bmi = tool_args.get("bmi")
            if not isinstance(age, int) or age < 1 or age > 120:
                errors["age"] = "年龄需为1-120的整数"
            if not isinstance(bmi, (int, float)) or bmi < 10 or bmi > 80:
                errors["bmi"] = "BMI需为10-80之间的数值"
        return errors

    def check_tool_needed_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: CHECK_TOOL_NEEDED - 技能选择与工具门控")
        state["need_tool"] = False
        state["selected_skill"] = ""
        state["selected_tool"] = ""
        state["tool_missing_params"] = []
        state["tool_selection_reason"] = ""
        if not self.tools:
            return state

        messages = state.get("messages", [])
        if not messages:
            return state

        user_question = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        if not self._is_assessment_skill_candidate(user_question):
            self.logger.info("未命中评估技能轻量门控，跳过技能激活")
            return state

        available_profile_lines = []
        for tool_name, profile in self.assessment_profiles.items():
            if tool_name in self.tool_map:
                available_profile_lines.append(
                    f"- {tool_name}({profile['display_name']}), 必需参数: {', '.join(profile['required_params'])}"
                )
        profiles_text = "\n".join(available_profile_lines)
        if not profiles_text:
            return state
        try:
            prompt = f"""你是健康风险评估技能路由器。请判断是否激活评估技能，并选择最匹配的工具。
可用评估工具:
{profiles_text}
用户问题: {user_question}
要求:
1) 只有当用户明确要风险评估/计算，或提供了可评估参数时，use_assessment_skill=true
2) selected_tool只能从可用评估工具里选择
3) missing_params只包含必需参数缺失项
4) 如果不激活技能，selected_tool返回null"""
            structured_llm = self.llm.with_structured_output(ToolSkillDecision)
            decision = structured_llm.invoke(prompt)
            selected_tool = (decision.selected_tool or "").strip()
            if selected_tool and selected_tool not in self.tool_map:
                selected_tool = ""
            if not selected_tool:
                if any(keyword in user_question for keyword in ["高血压", "血压", "收缩压", "舒张压", "高压", "低压"]):
                    selected_tool = "hypertension_risk_assessment" if "hypertension_risk_assessment" in self.tool_map else ""
                elif any(keyword in user_question.lower() for keyword in ["糖尿病", "血糖", "bmi", "体重指数"]):
                    selected_tool = "diabetes_risk_assessment" if "diabetes_risk_assessment" in self.tool_map else ""
            missing_params = list(decision.missing_params or [])
            if selected_tool:
                inferred_missing = self._missing_required_params(selected_tool, user_question)
                missing_params = list(dict.fromkeys([*missing_params, *inferred_missing]))
            state["selected_skill"] = self.assessment_skill_name if selected_tool else ""
            state["selected_tool"] = selected_tool
            state["tool_missing_params"] = missing_params
            state["tool_selection_reason"] = decision.reasoning
            state["need_tool"] = bool(state["selected_skill"] and state["selected_tool"])
            self.logger.info(
                f"技能路由: need_tool={state['need_tool']}, skill={state['selected_skill']}, tool={state['selected_tool']}, missing={missing_params}"
            )
        except Exception as e:
            self.logger.error(f"技能路由失败: {e}")
            state["need_tool"] = False
        return state

    def tool_calling_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        self.logger.info("="*50)
        self.logger.info("[RAG Graph] 节点: TOOL_CALLING - 执行工具调用")
        messages = state.get("messages", [])
        user_question = messages[-1].content if messages and hasattr(messages[-1], "content") else ""
        selected_skill = state.get("selected_skill", "")
        selected_tool = state.get("selected_tool")
        self.logger.info(f"用户问题: {user_question}")
        self.logger.info(f"选中的技能: {selected_skill}")
        self.logger.info(f"选中的工具: {selected_tool}")
        missing_params = state.get("tool_missing_params", [])
        if missing_params:
            answer = self._build_missing_params_answer(selected_tool or "", missing_params)
            state["final_answer"] = answer
            state["messages"] = [AIMessage(content=answer)]
            return state
        if not selected_tool or selected_tool not in self.tool_map:
            available_tools = list(self.tool_map.keys())
            error = ToolNotFoundError(tool_name=selected_tool or "unknown", available_tools=available_tools)
            state["error"] = error.to_dict()
            state["final_answer"] = error.message
            state["messages"] = [AIMessage(content=error.message)]
            return state
        bound_tool = self.tool_map[selected_tool]
        required_params = self.assessment_profiles.get(selected_tool, {}).get("required_params", [])
        required_params_text = ", ".join(required_params) if required_params else "无"
        try:
            llm_with_tools = self.llm.bind_tools([bound_tool])
            prompt = f"""你是健康评估技能执行器。
用户问题：{user_question}
已选择技能：{selected_skill}
你必须且只能调用工具：{selected_tool}
必需参数：{required_params_text}
请从用户问题中抽取参数并调用工具。"""
            self.logger.info("调用LLM，期待工具调用...")
            response = llm_with_tools.invoke(prompt)
            if hasattr(response, 'tool_calls') and response.tool_calls:
                self.logger.info(f"LLM请求调用 {len(response.tool_calls)} 个工具")
                audit_logger = get_audit_logger()
                target_calls = [call for call in response.tool_calls if call.get("name") == selected_tool]
                if not target_calls:
                    state["final_answer"] = f"评估执行失败：模型未按要求调用 {selected_tool}。请重试或补充更明确参数。"
                    state["messages"] = [AIMessage(content=state["final_answer"])]
                    return state
                selected_call = target_calls[0]
                tool_args = selected_call.get("args", {})
                validation_errors = self._validate_tool_args(selected_tool, tool_args)
                if validation_errors:
                    error = ToolValidationError(
                        tool_name=selected_tool,
                        validation_errors=validation_errors,
                        details="工具参数预校验失败"
                    )
                    state["error"] = error.to_dict()
                    state["final_answer"] = f"{error.message}，请补充或修正参数后重试。"
                    state["messages"] = [AIMessage(content=state["final_answer"])]
                    return state
                start_time = time.time()
                tool_result = None
                tool_error = None
                try:
                    tool_result = self._execute_tool_with_timeout(bound_tool, tool_args)
                    self.logger.info(f"工具执行成功: {selected_tool}")
                except ToolExecutionTimeoutError as timeout_error:
                    tool_error = timeout_error.to_dict()
                    state["error"] = tool_error
                    state["final_answer"] = timeout_error.message
                    state["messages"] = [AIMessage(content=timeout_error.message)]
                    return state
                except Exception as tool_exec_error:
                    error = ToolExecutionError(
                        tool_name=selected_tool,
                        error_message=str(tool_exec_error),
                        details=f"{type(tool_exec_error).__name__}: {tool_exec_error}"
                    )
                    tool_error = error.to_dict()
                    state["error"] = tool_error
                    state["final_answer"] = error.message
                    state["messages"] = [AIMessage(content=error.message)]
                    return state
                finally:
                    execution_time_ms = (time.time() - start_time) * 1000
                    audit_logger.log_tool_call(
                        conversation_id=state.get("session_id", "unknown"),
                        user_id=state.get("user_id", "unknown"),
                        tool_name=selected_tool,
                        tool_args=tool_args,
                        result=tool_result,
                        error=tool_error,
                        execution_time_ms=execution_time_ms,
                        metadata={"selected_skill": selected_skill, "selection_reason": state.get("tool_selection_reason", "")}
                    )
                final_prompt = f"""基于以下工具执行结果，请给用户一个清晰、专业、友好的回答。
用户问题：{user_question}
工具结果：
{json.dumps(tool_result, ensure_ascii=False, indent=2)}
请用通俗语言解释结论并给出可执行建议。"""
                final_response = self.llm.invoke(final_prompt)
                state["final_answer"] = final_response.content
                state["answer_sources"] = [{
                    "index": 1,
                    "tool_name": selected_tool,
                    "document_name": f"工具: {selected_tool}",
                    "content": f"风险等级: {tool_result.get('risk_level', 'N/A')}, 评分: {tool_result.get('risk_score', 'N/A')}",
                    "retrieval_mode": "tool",
                    "metadata": {
                        "tool_args": tool_args,
                        "risk_level": tool_result.get("risk_level"),
                        "risk_score": tool_result.get("risk_score"),
                        "risk_factors": tool_result.get("risk_factors", []),
                        "recommendations": tool_result.get("recommendations", []),
                        "selected_skill": selected_skill
                    }
                }]
                state["messages"] = [AIMessage(content=final_response.content)]
                return state
            state["final_answer"] = "抱歉，我无法从您的问题中提取足够的参数来进行评估。请提供更详细的信息。"
            state["messages"] = [AIMessage(content=state["final_answer"])]
        except ToolCallException as e:
            self.logger.error(f"工具调用异常: {e.to_dict()}")
            state["error"] = e.to_dict()
            state["final_answer"] = e.message
            state["messages"] = [AIMessage(content=e.message)]
        except Exception as e:
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
            state["retrieval_mode"] = RetrievalMode.VECTOR_ONLY
            state["retrieval_mode_reason"] = "AUTO模式默认向量检索"

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
                    retrieval_type = (decision.retrieval_type or "").strip().lower()
                    # 根据LLM判断结果更新检索模式
                    if retrieval_type == "vector_only":
                        state["retrieval_mode"] = RetrievalMode.VECTOR_ONLY
                        self.logger.info(f"LLM判断：使用向量检索 - {decision.reasoning}")
                    elif retrieval_type == "hybrid":
                        state["retrieval_mode"] = RetrievalMode.HYBRID
                        self.logger.info(f"LLM判断：使用融合检索 - {decision.reasoning}")
                    elif retrieval_type == "graph_only":
                        state["retrieval_mode"] = RetrievalMode.HYBRID
                        self.logger.info(f"LLM判断：graph_only已映射为融合检索 - {decision.reasoning}")
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

    def _deduplicate_retrieved_docs(self, docs: list[RetrievedDocument]) -> list[RetrievedDocument]:
        unique_docs = []
        seen_keys = set()
        for doc in docs:
            metadata = doc.metadata or {}
            pk = metadata.get("pk") or metadata.get("id")
            if pk is not None:
                key = ("pk", str(pk))
            else:
                document_name = metadata.get("document_name", "")
                chunk_index = metadata.get("chunk_index", "")
                key = ("fallback", str(document_name), str(chunk_index), doc.page_content[:80])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_docs.append(doc)
        return unique_docs

    def _is_near_duplicate(self, left: str, right: str) -> bool:
        if not left or not right:
            return False
        normalized_left = "".join(ch.lower() for ch in left if not ch.isspace())
        normalized_right = "".join(ch.lower() for ch in right if not ch.isspace())
        if not normalized_left or not normalized_right:
            return False
        if normalized_left[:180] == normalized_right[:180]:
            return True
        left_set = set(normalized_left[:800])
        right_set = set(normalized_right[:800])
        if not left_set or not right_set:
            return False
        overlap = len(left_set & right_set)
        union = len(left_set | right_set)
        if union == 0:
            return False
        return (overlap / union) >= 0.92

    def _deduplicate_semantic_docs(self, docs: list[RetrievedDocument]) -> list[RetrievedDocument]:
        deduped_docs: list[RetrievedDocument] = []
        for doc in docs:
            is_dup = False
            current_name = (doc.metadata or {}).get("document_name", "")
            for existing_doc in deduped_docs:
                existing_name = (existing_doc.metadata or {}).get("document_name", "")
                if current_name and existing_name and current_name != existing_name:
                    continue
                if self._is_near_duplicate(doc.page_content, existing_doc.page_content):
                    is_dup = True
                    break
            if not is_dup:
                deduped_docs.append(doc)
        return deduped_docs

    def _extract_overlap_score(self, query_text: str, content: str) -> float:
        if not query_text or not content:
            return 0.0
        query_tokens = set(ch.lower() for ch in query_text if not ch.isspace())
        content_tokens = set(ch.lower() for ch in content[:1000] if not ch.isspace())
        if not query_tokens:
            return 0.0
        overlap = len(query_tokens & content_tokens)
        return overlap / len(query_tokens)

    def _content_similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        left_tokens = set(ch.lower() for ch in left[:1000] if not ch.isspace())
        right_tokens = set(ch.lower() for ch in right[:1000] if not ch.isspace())
        if not left_tokens or not right_tokens:
            return 0.0
        inter = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        if union == 0:
            return 0.0
        return inter / union

    def _mmr_select_docs(self, docs: list[RetrievedDocument], select_k: int) -> list[RetrievedDocument]:
        if select_k <= 0 or not docs:
            return []
        candidate_docs = list(docs)
        selected_docs: list[RetrievedDocument] = []
        while candidate_docs and len(selected_docs) < select_k:
            best_doc = None
            best_score = None
            for doc in candidate_docs:
                relevance = float((doc.metadata or {}).get("rrf_score", 0.0))
                if not selected_docs:
                    mmr_score = relevance
                else:
                    redundancy = max(
                        self._content_similarity(doc.page_content, selected.page_content)
                        for selected in selected_docs
                    )
                    mmr_score = self.mmr_lambda * relevance - (1.0 - self.mmr_lambda) * redundancy
                if best_score is None or mmr_score > best_score:
                    best_score = mmr_score
                    best_doc = doc
            if best_doc is None:
                break
            selected_docs.append(best_doc)
            candidate_docs.remove(best_doc)
        return selected_docs

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot_value = sum(l * r for l, r in zip(left, right))
        left_norm = math.sqrt(sum(l * l for l in left))
        right_norm = math.sqrt(sum(r * r for r in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot_value / (left_norm * right_norm)

    def _semantic_rerank_docs(self, query_text: str, docs: list[RetrievedDocument]) -> list[RetrievedDocument]:
        if not self.enable_semantic_rerank or not self.embedding_model or not query_text or not docs:
            return docs
        try:
            query_vector = self.embedding_model.embed_query(query_text)
            doc_texts = [doc.page_content for doc in docs]
            doc_vectors = self.embedding_model.embed_documents(doc_texts)
            scored_docs = []
            for doc, vector in zip(docs, doc_vectors):
                semantic_score = self._cosine_similarity(query_vector, vector)
                metadata = dict(doc.metadata or {})
                metadata["semantic_score"] = round(semantic_score, 6)
                scored_docs.append(
                    RetrievedDocument(
                        page_content=doc.page_content,
                        metadata=metadata
                    )
                )
            scored_docs.sort(
                key=lambda d: (
                    float((d.metadata or {}).get("semantic_score", 0.0)),
                    float((d.metadata or {}).get("rrf_score", 0.0))
                ),
                reverse=True
            )
            return scored_docs
        except Exception as exc:
            self.logger.warning(f"语义重排失败，回退至RRF排序: {exc}")
            return docs

    def _split_graph_result_to_docs(self, content: str, max_docs: int) -> list[RetrievedDocument]:
        stripped = (content or "").strip()
        if not stripped:
            return []
        raw_segments = [seg.strip() for seg in stripped.split("\n\n") if seg.strip()]
        if not raw_segments:
            raw_segments = [stripped]
        docs: list[RetrievedDocument] = []
        limit = max(max_docs * 2, 4)
        for idx, segment in enumerate(raw_segments[:limit]):
            docs.append(
                RetrievedDocument(
                    page_content=segment,
                    metadata={
                        "source": "lightrag_graph",
                        "retrieval_mode": "global",
                        "document_name": "知识图谱检索结果",
                        "chunk_index": idx,
                        "rank_graph": idx + 1
                    }
                )
            )
        return docs

    def _merge_retrieved_docs(
        self,
        vector_docs: list[RetrievedDocument],
        graph_docs: list[RetrievedDocument],
        max_docs: int,
        query_text: str
    ) -> list[RetrievedDocument]:
        rrf_k = self.rrf_k
        source_weights = {
            "vector": self.vector_source_weight,
            "graph": self.graph_source_weight
        }
        score_map: dict[tuple, float] = {}
        doc_map: dict[tuple, RetrievedDocument] = {}
        source_map: dict[tuple, set[str]] = {}

        def doc_key(doc: RetrievedDocument) -> tuple:
            metadata = doc.metadata or {}
            pk = metadata.get("pk") or metadata.get("id")
            if pk is not None:
                return ("pk", str(pk))
            return (
                "fallback",
                str(metadata.get("document_name", "")),
                str(metadata.get("chunk_index", "")),
                doc.page_content[:120]
            )

        def add_with_rrf(docs: list[RetrievedDocument], source_tag: str) -> None:
            for rank_index, doc in enumerate(docs or []):
                key = doc_key(doc)
                score = source_weights[source_tag] * (1.0 / (rrf_k + rank_index + 1))
                lexical_bonus = 0.08 * self._extract_overlap_score(query_text, doc.page_content)
                total = score + lexical_bonus
                score_map[key] = score_map.get(key, 0.0) + total
                if key not in doc_map:
                    doc_map[key] = doc
                source_set = source_map.get(key, set())
                source_set.add(source_tag)
                source_map[key] = source_set

        add_with_rrf(vector_docs, "vector")
        add_with_rrf(graph_docs, "graph")

        ranked_items = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
        ranked_docs: list[RetrievedDocument] = []
        for rank_idx, (key, score) in enumerate(ranked_items):
            doc = doc_map[key]
            metadata = dict(doc.metadata or {})
            metadata["rrf_score"] = round(score, 6)
            metadata["fused_rank"] = rank_idx + 1
            metadata["source"] = "hybrid_fused"
            metadata["fused_sources"] = sorted(list(source_map.get(key, set())))
            ranked_docs.append(RetrievedDocument(page_content=doc.page_content, metadata=metadata))

        ranked_docs = self._deduplicate_retrieved_docs(ranked_docs)
        ranked_docs = self._deduplicate_semantic_docs(ranked_docs)
        ranked_docs = self._semantic_rerank_docs(query_text, ranked_docs)

        text_docs = [doc for doc in ranked_docs if (doc.metadata or {}).get("chunk_type") != "chart"]
        chart_docs = [doc for doc in ranked_docs if (doc.metadata or {}).get("chunk_type") == "chart"]

        vector_text_docs = []
        graph_text_docs = []
        neutral_text_docs = []
        for doc in text_docs:
            fused_sources = set((doc.metadata or {}).get("fused_sources") or [])
            if "graph" in fused_sources and "vector" in fused_sources:
                neutral_text_docs.append(doc)
            elif "graph" in fused_sources:
                graph_text_docs.append(doc)
            else:
                vector_text_docs.append(doc)

        candidate_text_docs: list[RetrievedDocument] = []
        while vector_text_docs or graph_text_docs:
            if vector_text_docs:
                candidate_text_docs.append(vector_text_docs.pop(0))
            if graph_text_docs:
                candidate_text_docs.append(graph_text_docs.pop(0))
        candidate_text_docs.extend(neutral_text_docs)
        selected_text_docs = self._mmr_select_docs(candidate_text_docs, max_docs)

        max_chart_docs = 2 if max_docs >= 2 else 1
        selected_chart_docs = chart_docs[:max_chart_docs]

        fused_docs = selected_text_docs + selected_chart_docs
        if not fused_docs:
            fused_docs = ranked_docs[:max_docs]
        return fused_docs

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
                search_kwargs={"k": max(max_docs * 2, 6)}
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
                metadata = dict(doc.metadata or {})
                metadata["source"] = metadata.get("source") or "vector"
                retrieved_doc = RetrievedDocument(
                    page_content=doc.page_content,
                    metadata=metadata
                )
                converted_docs.append(retrieved_doc)
                # 调试日志：查看每个文档的metadata
                self.logger.info(f"检索文档 metadata: document_name={doc.metadata.get('document_name')}, "
                               f"chunk_index={doc.metadata.get('chunk_index')}, "
                               f"content_length={len(doc.page_content)}, "
                               f"pk={doc.metadata.get('pk')}")

            unique_docs = self._deduplicate_retrieved_docs(converted_docs)

            self.logger.info(f"去重后文档数: {len(unique_docs)}")

            text_docs = [doc for doc in unique_docs if doc.metadata.get("chunk_type") != "chart"]
            chart_docs = [doc for doc in unique_docs if doc.metadata.get("chunk_type") == "chart"]

            max_chart_docs = 2 if max_docs >= 2 else 1
            selected_text_docs = text_docs[:max_docs]
            selected_chart_docs = chart_docs[:max_chart_docs]
            fused_docs = selected_text_docs + selected_chart_docs
            if not fused_docs:
                fused_docs = unique_docs[:max_docs]

            self.logger.info(
                f"融合检索结果: 文本 {len(selected_text_docs)} 条, 图表 {len(selected_chart_docs)} 条, 总计 {len(fused_docs)} 条"
            )

            state["retrieved_docs"] = fused_docs
            state["vector_db_results"] = unique_docs

        except Exception as e:
            self.logger.error(f"向量检索失败: {e}")
            # 检索失败时设置空结果
            state["retrieved_docs"] = []
            state["vector_db_results"] = []

        return state

    async def hybrid_retrieval_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: HYBRID_RETRIEVAL - 融合检索")

        context = runtime.context
        max_docs = context.max_retrieval_docs if context else 3

        state = self.vector_db_retrieval_node(state, runtime)
        vector_docs = list(state.get("vector_db_results") or [])

        state = await self.graph_db_retrieval_node(state, runtime)
        graph_docs = list(state.get("graph_db_results") or [])

        query_text = state.get("original_question", "")
        merged_docs = self._merge_retrieved_docs(vector_docs, graph_docs, max_docs, query_text)
        state["retrieved_docs"] = merged_docs
        state["vector_db_results"] = vector_docs
        state["graph_db_results"] = graph_docs
        state["retrieval_fusion_stats"] = {
            "vector_docs": len(vector_docs),
            "graph_docs": len(graph_docs),
            "merged_docs": len(merged_docs),
            "rrf_k": self.rrf_k,
            "mmr_lambda": self.mmr_lambda
        }

        self.logger.info(
            f"融合检索完成: vector={len(vector_docs)}, graph={len(graph_docs)}, merged={len(merged_docs)}"
        )
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

                context = runtime.context
                max_docs = context.max_retrieval_docs if context else 3
                graph_docs = self._split_graph_result_to_docs(result, max_docs)

                state["retrieved_docs"] = graph_docs
                state["graph_db_results"] = graph_docs

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
                chart_image_url = doc.metadata.get("chart_image_url")
                if chart_image_url:
                    try:
                        parsed = urlparse(chart_image_url.strip().strip("`"))
                        host = parsed.netloc or ""
                        if "aliyuncs.com" in host:
                            path = (parsed.path or "").lstrip("/")
                            bucket = host.split(".")[0] if host else None
                            if not bucket or bucket.startswith("oss"):
                                if "/" in path:
                                    bucket, path = path.split("/", 1)
                            key = unquote(path)
                            if bucket and key:
                                presign = get_presigned_url_for_download(bucket=bucket, key=key)
                                if presign and presign.get("url"):
                                    chart_image_url = presign.get("url")
                    except Exception:
                        pass

                source_info = {
                    "index": i + 1,
                    "document_name": doc.metadata.get("document_name", f"文档{i+1}"),
                    "content": content_preview,
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "retrieval_mode": retrieval_source,
                    "content_length": len(doc.page_content),
                    "chunk_type": doc.metadata.get("chunk_type", "text"),
                    "chart_id": doc.metadata.get("chart_id"),
                    "chart_image_url": chart_image_url,
                    "section_title": doc.metadata.get("section_title"),
                    "page_number": doc.metadata.get("page_number")
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
        elif retrieval_mode == RetrievalMode.HYBRID:
            return "hybrid_db"
        elif retrieval_mode == RetrievalMode.GRAPH_ONLY:
            return "graph_db"
        elif retrieval_mode == RetrievalMode.AUTO:
            # AUTO模式默认使用向量检索，可以根据需要扩展智能判断逻辑
            return "vector_db"
        else:
            # 默认使用向量数据库
            return "vector_db"
