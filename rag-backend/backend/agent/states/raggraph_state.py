from typing import Dict, Any, List, TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from ..contexts.raggraph_context import RAGContext
from ..models.raggraph_models import RetrievedDocument


class RAGGraphState(TypedDict, total=False):
    """RAG图状态管理
    
    使用TypedDict定义状态结构，确保与LangGraph兼容
    total=False表示所有字段都是可选的
    """
    # ==================== 核心字段 ====================
    messages: Annotated[List[BaseMessage], add_messages]  # 消息历史（LangGraph核心字段）
    
    # ==================== 基础信息 ====================
    session_id: str                    # 会话ID
    user_id: str                       # 用户ID（从context获取）
    original_question: str             # 原始问题
    # ==================== Context相关 ====================
    retrieval_config: Dict[str, Any]   # 检索配置（从context获取）
    system_prompt: str                 # 系统提示（从context获取）
    
    # ==================== 流程控制 ====================
    retrieval_mode: str                # 检索模式（vector_only/hybrid/graph_only/no_retrieval/auto）
    need_retrieval: bool               # 是否需要检索
    need_retrieval_reason: Optional[str]    # 需要检索的理由
    retrieval_decision_stats: Dict[str, Any]  # 检索决策链路统计
    retrieval_mode_reason: Optional[str]    # 检索模式的理由
    need_tool: bool                    # 是否需要调用工具
    selected_tool: Optional[str]        # 选中的工具名称
    selected_skill: Optional[str]       # 选中的技能名称
    tool_missing_params: List[str]           # 工具调用缺失参数
    tool_prefilled_args: Dict[str, Any]      # 从历史消息中提取的工具参数
    tool_selection_reason: Optional[str]  # 工具/技能选择理由
    tool_clarify_message: Optional[str]
    pending_tool_name: Optional[str]
    pending_tool_deadline_ms: int
    # ==================== 医疗SOP链路 ====================
    medical_structured_output: Dict[str, Any]  # 结构化医疗决策输出
    extracted_symptoms: List[Dict[str, Any]]   # 抽取的症状信息
    extracted_vitals: Dict[str, Any]           # 抽取的体征信息
    patient_profile_snapshot: Dict[str, Any]   # 患者画像快照
    medical_red_flags: List[str]               # 医疗红线命中项
    triage_level: str                          # 分诊等级（emergency/urgent/routine/unknown）
    handoff_required: bool                     # 是否触发转人工
    handoff_reason: Optional[str]              # 转人工原因
    intervention_plan: Dict[str, Any]          # 个性化干预草案
    structured_decision_valid: bool            # 结构化决策是否有效
    structured_decision_error: Optional[str]   # 结构化决策错误信息
    # ==================== 问题处理 ====================
    subquestions: List[str]            # 扩展的子问题列表
    subquestion_expansion_stats: Dict[str, Any]  # 子问题扩展统计
    processed_questions: List[str]     # 已处理的问题列表
    
    # ==================== 检索结果 ====================
    retrieved_docs: List[RetrievedDocument]  # 检索到的文档列表
    vector_db_results: List[RetrievedDocument]  # 向量数据库检索结果
    graph_db_results: List[RetrievedDocument]   # 图数据库检索结果
    graph_retrieval_stats: Dict[str, Any]  # 图检索统计信息
    retrieval_fusion_stats: Dict[str, Any]  # 融合检索统计信息
    
    # ==================== 答案生成 ====================
    final_answer: str                  # 最终答案
    answer_sources: List[Dict[str, Any]]  # 答案来源列表
    
    # ==================== 错误处理 ====================
    error: Optional[Dict[str, Any]]  # 结构化错误信息


def create_initial_rag_state(
    context: RAGContext,
    input_data: Dict[str, Any],
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> RAGGraphState:
    """创建初始化的RAGGraphState
    
    Args:
        context: RAG上下文配置
        input_data: 输入数据，包含messages等字段
        session_id: 会话ID，如果为None则使用context中的session_id
        user_id: 用户ID，如果为None则使用context中的user_id
        
    Returns:
        初始化的RAGGraphState实例
    """
    return RAGGraphState(
        # ==================== 核心字段 ====================
        messages=input_data.get("messages", []),
        
        # ==================== 基础信息 ====================
        session_id=session_id or context.session_id or "default",
        user_id=user_id or context.user_id or "anonymous",
        
        
        # ==================== Context相关 ====================
        retrieval_config=context.get_retrieval_config(),
        system_prompt=context.get_system_prompt(),
        
        # ==================== 流程控制 ====================
        retrieval_mode=context.retrieval_mode,
        need_retrieval=True,
        need_retrieval_reason="",
        retrieval_decision_stats={},
        need_tool=False,
        selected_tool="",
        selected_skill="",
        tool_missing_params=[],
        tool_prefilled_args={},
        tool_selection_reason="",
        tool_clarify_message="",
        pending_tool_name="",
        pending_tool_deadline_ms=0,
        medical_structured_output={},
        extracted_symptoms=[],
        extracted_vitals={},
        patient_profile_snapshot={},
        medical_red_flags=[],
        triage_level="unknown",
        handoff_required=False,
        handoff_reason="",
        intervention_plan={},
        structured_decision_valid=False,
        structured_decision_error="",
        
        # ==================== 问题处理 ====================
        original_question="",
        subquestions=[],
        subquestion_expansion_stats={},
        processed_questions=[],
        
        # ==================== 检索结果 ====================
        retrieved_docs=[],
        vector_db_results=[],
        graph_db_results=[],
        graph_retrieval_stats={},
        retrieval_fusion_stats={},
        
        # ==================== 答案生成 ====================
        final_answer="",
        answer_sources=[]
    )
    
