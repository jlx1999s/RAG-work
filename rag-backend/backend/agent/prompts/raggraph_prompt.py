"""RAG Graph 相关提示词管理

本模块包含RAG图中各个节点使用的提示词模板。
"""

from typing import Dict, Any
from pydantic import BaseModel, Field
from typing import List


class RetrievalNeedDecision(BaseModel):
    """检索需求判断结果"""
    need_retrieval: bool = Field(description="是否需要进行检索")
    extracted_question: str = Field(description="提取的核心问题")
    reasoning: str = Field(description="判断是否需要检索的理由")


class SubquestionExpansion(BaseModel):
    """子问题扩展结果"""
    subquestions: List[str] = Field(description="扩展出的子问题列表")


class RetrievalTypeDecision(BaseModel):
    """检索类型判断结果"""
    retrieval_type: str = Field(description="推荐的检索类型：vector_only、hybrid或graph_only")
    reasoning: str = Field(description="选择该检索类型的理由")

class ToolSkillDecision(BaseModel):
    use_assessment_skill: bool = Field(description="是否需要进入健康评估技能流程")
    selected_tool: str | None = Field(description="命中的评估工具名，如hypertension_risk_assessment")
    reasoning: str = Field(description="选择该技能或工具的理由")
    missing_params: List[str] = Field(default_factory=list, description="缺失的必需参数列表")


class MedicalStructuredDecision(BaseModel):
    """结构化医疗SOP决策结果"""
    is_medical: bool = Field(description="是否属于医疗相关问题")
    symptoms: List[str] = Field(default_factory=list, description="提取到的症状列表")
    vitals: Dict[str, Any] = Field(default_factory=dict, description="提取到的体征信息，如收缩压、舒张压、体温等")
    red_flags: List[str] = Field(default_factory=list, description="急危重症红线命中项")
    triage_level: str = Field(default="routine", description="分诊等级：emergency/urgent/routine/unknown")
    handoff_required: bool = Field(default=False, description="是否需要立即转人工或急诊")
    handoff_reason: str = Field(default="", description="触发转人工原因")
    intervention_focus: List[str] = Field(default_factory=list, description="后续干预建议关注点")
    reasoning: str = Field(default="", description="结构化决策理由")


class MedicalInterventionPlan(BaseModel):
    """结构化个性化干预草案"""
    summary: str = Field(default="", description="干预方案摘要")
    lifestyle_actions: List[str] = Field(default_factory=list, description="生活方式建议")
    followup_actions: List[str] = Field(default_factory=list, description="随访与复诊建议")
    escalation_signals: List[str] = Field(default_factory=list, description="需要及时就医的危险信号")
    medication_cautions: List[str] = Field(default_factory=list, description="用药注意事项")
    reasoning: str = Field(default="", description="干预规划理由")


class RAGGraphPrompts:
    """RAG Graph 提示词集合"""

    @staticmethod
    def get_direct_answer_prompt() -> str:
        """获取直接回答节点的提示词（支持短期对话记忆）
        
        Returns:
            直接回答的提示词模板
        """
        return """你是一个专业且友好的AI助手。下面是你和用户的对话历史（时间从早到晚）：

{conversation_history}

现在用户的最新问题是：
{question}

请结合上面的对话历史，提供准确、清晰的回答。如果问题不清楚，可以要求用户提供更多信息；如果用户之前已经给出相关信息（如名字、偏好），请保持前后一致。"""

    @staticmethod
    def get_direct_answer_memory_prompt() -> str:
        """获取直接回答节点的记忆管理提示词（已弃用）
        """
        return f"""你是一个智能助手，拥有长期记忆功能。
    回答任何问题前一定要先对记忆进行回忆。
在回答用户问题前，请先调用 search_memory 工具，检查是否有相关记忆。
然后再决定是否需要调用 manage_memory 更新记忆。"""

    @staticmethod
    def get_retrieval_need_judgment_prompt() -> str:
        """获取判断是否需要检索的提示词
        
        Returns:
            判断检索需求的提示词模板
        """
        return """
你是一个智能的检索需求判断助手。你的任务是分析用户的问题，判断是否需要进行外部知识检索来回答这个问题。

请根据以下标准进行判断：

**需要检索的情况：**
1. 问题涉及具体的事实信息、数据、统计数字
2. 问题询问特定的人物、事件、地点、组织等
3. 问题需要最新的信息或时事
4. 问题涉及专业领域的具体知识
5. 问题询问具体的产品、服务、技术细节
6. 问题需要引用权威资料或文档

**不需要检索的情况：**
1. 纯粹的数学计算或逻辑推理
2. 一般性的概念解释或常识问题
3. 创意性问题（如写作、头脑风暴）
4. 个人观点或主观判断
5. 简单的定义或解释
6. 程序代码编写或调试（基础层面）

**用户问题：**
{question}

请分析这个问题，判断是否需要进行检索，并提取出核心问题。

**请按照以下格式输出你的判断结果：**

need_retrieval: [true/false] - 是否需要进行检索
extracted_question: [提取的核心问题] - 从用户问题中提取出的关键问题
reasoning: [判断理由] - 详细说明为什么需要或不需要检索的原因

确保你的回答准确反映问题的性质，并提供清晰的判断依据。

"""
    
    @staticmethod
    def get_retrieval_type_judgment_prompt() -> str:
        """获取判断检索类型的提示词
        
        Returns:
            判断检索类型的提示词模板
        """
        return """
你是一个智能的检索类型判断助手。你的任务是分析用户的问题，判断应该使用向量检索还是融合检索来获取最相关的信息。

**向量检索 (vector_only) 适用场景：**
1. 语义相似性查询：寻找意思相近的内容
2. 模糊匹配：关键词不完全匹配但语义相关
3. 概念性问题：需要理解概念含义的查询
4. 文本内容检索：主要基于文档内容进行匹配
5. 跨语言查询：不同语言但相同含义的查询
6. 长文本查询：复杂的自然语言描述

**融合检索 (hybrid) 适用场景：**
1. 关系查询：询问实体之间的关系
2. 路径查询：需要通过多个节点找到答案
3. 结构化查询：基于明确的实体和属性
4. 精确匹配：需要准确的实体或属性值
5. 层次查询：涉及分类、层级关系的问题
6. 连接查询：需要连接多个相关实体的信息

默认优先 vector_only，仅在问题明显复杂或需要关系推理时返回 hybrid。graph_only 仅在问题明确要求图谱关系且不需要向量语义补充时返回。

**用户问题：**
{question}

请分析这个问题的特点，判断应该使用哪种检索方式，并说明理由。

请按照以下JSON格式返回结果：
{{
    "retrieval_type": "vector_only" 或 "hybrid" 或 "graph_only",
    "reasoning": "选择该检索类型的详细理由"
}}
"""
    
    @staticmethod
    def get_subquestion_expansion_prompt() -> str:
        """获取子问题扩展提示词"""
        return """
你是一个专业的问题分析助手。请根据用户的原始问题，将其分解为多个具体的子问题，以便更好地进行信息检索和回答。

分解原则：
1. 子问题应该涵盖原始问题的各个方面
2. 每个子问题应该具体明确，便于检索
3. 子问题之间应该相互补充，避免重复
4. 子问题数量通常在1-3个之间
5. 如果原始问题已经足够具体，可以只生成1个子问题（即原问题本身）

原始问题：{question}

请将上述问题分解为具体的子问题列表。
"""
    

    
    @staticmethod
    def get_answer_generation_prompt() -> str:
        """获取答案生成的提示词
        
        Returns:
            答案生成的提示词模板
        """
        return """
你是一个严格遵循上下文的问答助手，只能使用给定上下文作答。

**回答要求：**
1. 只使用给定上下文内容，不引入任何外部知识或常识
2. 提供信息来源和参考依据
3. 不解释、不对比、不延伸、不补充背景
4. 不使用表格，不做大段总结，只回答问题本身
5. 如果上下文中没有答案，只回复：未在给定上下文中找到答案
6. 输出简洁、直接，避免多余措辞

**用户问题：**
{question}

**给定上下文（共{doc_count}个）：**
{documents}

"""

    @staticmethod
    def get_conversation_summary_prompt() -> str:
        """获取对话摘要生成提示词
        
        用于将同一会话中较早的多轮对话压缩为简洁摘要，
        摘要仅在当前 conversation_id 内使用，不跨对话共享。
        """
        return """
你是一个对话总结助手。下面是一段同一会话中较早的多轮对话内容，请你用简洁的中文总结：
- 用户的大致身份或角色（如果有提到）
- 本轮对话的主要目标或主题
- 已经达成的重要结论或共识
不要逐句复述原文，只保留对后续对话有用的关键信息，控制在 3~5 句话之内。

【对话历史】
{history}
"""

    @staticmethod
    def get_medical_sop_prompt() -> str:
        """获取结构化医疗SOP决策提示词"""
        return """
你是医疗SOP执行助手。请严格按JSON输出，不要输出任何解释文本。

你必须遵循以下顺序：
1) 体征与症状抽取
2) 医疗红线检查（急危重症拦截与转人工）
3) 个性化干预关注点生成

用户问题：
{question}

患者画像快照（可能为空）：
{patient_profile}

规则：
- 若命中急危重症（如胸痛、呼吸困难、意识障碍、抽搐、活动性出血等），handoff_required必须为true，triage_level必须为emergency。
- 若信息不足但存在明显风险，也应设置handoff_required=true，并给出handoff_reason。
- 不得给出处方级、诊断结论级的建议；只给出结构化判定结果。

请输出字段：
- is_medical
- symptoms
- vitals
- red_flags
- triage_level
- handoff_required
- handoff_reason
- intervention_focus
- reasoning
"""

    @staticmethod
    def get_intervention_plan_prompt() -> str:
        """获取结构化个性化干预规划提示词"""
        return """
你是医疗干预规划助手。请基于已通过红线检查的结构化信息输出JSON，不要输出解释文本。

用户问题：
{question}

结构化SOP结果：
{medical_structured_output}

可用文档摘要：
{documents}

要求：
- 仅输出可执行且保守的健康管理建议
- 不输出确诊性结论，不输出处方级建议
- 对高风险信号给出明确的就医升级提醒

请输出字段：
- summary
- lifestyle_actions
- followup_actions
- escalation_signals
- medication_cautions
- reasoning
"""
