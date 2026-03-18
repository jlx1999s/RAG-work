import os
import sys
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from backend.agent.graph.raggraph_node import RAGNodes


class _DummyTool:
    def __init__(self, name: str):
        self.name = name

    def invoke(self, args):
        return {
            "risk_level": "low",
            "risk_score": 12,
            "risk_factors": [],
            "recommendations": [],
            "received_args": args,
        }


class _DummyLLM:
    def bind_tools(self, tools):
        return self

    def invoke(self, prompt):
        if "你是健康评估技能执行器" in prompt:
            return SimpleNamespace(
                tool_calls=[
                    {
                        "name": "hypertension_risk_assessment",
                        "args": {"systolic_bp": 120, "diastolic_bp": 88},
                    }
                ]
            )
        return SimpleNamespace(content="已生成回答")


def test_followup_bp_parameters_should_trigger_hypertension_tool():
    nodes = RAGNodes(
        llm=None,
        tools=[
            _DummyTool("hypertension_risk_assessment"),
            _DummyTool("diabetes_risk_assessment"),
        ],
    )
    state = {
        "messages": [
            HumanMessage(content="我年龄18、systolic_bp、diastolic_bp 120/88，帮我做风险评估"),
            AIMessage(content="要进行高血压风险评估，还需要补充以下参数：systolic_bp、diastolic_bp。请补充后我将自动继续评估。"),
            HumanMessage(content="systolic_bp120，diastolic_bp88"),
        ]
    }

    updated = nodes.check_tool_needed_node(state, runtime=None)

    assert updated["need_tool"] is True
    assert updated["selected_tool"] == "hypertension_risk_assessment"
    assert updated["tool_missing_params"] == []
    assert updated["tool_prefilled_args"]["age"] == 18
    assert updated["tool_prefilled_args"]["systolic_bp"] == 120
    assert updated["tool_prefilled_args"]["diastolic_bp"] == 88


def test_tool_calling_should_merge_prefilled_args_with_model_args():
    nodes = RAGNodes(
        llm=_DummyLLM(),
        tools=[_DummyTool("hypertension_risk_assessment")],
    )
    state = {
        "messages": [HumanMessage(content="systolic_bp120，diastolic_bp88")],
        "selected_skill": "health_risk_assessment",
        "selected_tool": "hypertension_risk_assessment",
        "tool_missing_params": [],
        "tool_selection_reason": "承接上一轮补参流程",
        "tool_prefilled_args": {"age": 18, "systolic_bp": 120, "diastolic_bp": 88},
        "session_id": "test_session",
        "user_id": "test_user",
    }

    output = nodes.tool_calling_node(state, runtime=None)

    assert output.get("error") is None
    assert output["answer_sources"][0]["metadata"]["tool_args"]["age"] == 18
    assert output["answer_sources"][0]["metadata"]["tool_args"]["systolic_bp"] == 120
    assert output["answer_sources"][0]["metadata"]["tool_args"]["diastolic_bp"] == 88


def test_pending_tool_should_not_hijack_irrelevant_question():
    nodes = RAGNodes(
        llm=None,
        tools=[
            _DummyTool("hypertension_risk_assessment"),
            _DummyTool("diabetes_risk_assessment"),
        ],
    )
    state = {
        "messages": [
            HumanMessage(content="我有没有糖尿病风险"),
            AIMessage(content="要进行糖尿病风险评估，还需要补充以下参数：age、bmi。请补充后我将自动继续评估。"),
            HumanMessage(content="发烧了怎么办"),
        ],
        "pending_tool_name": "diabetes_risk_assessment",
        "pending_tool_deadline_ms": 32503680000000,
    }
    updated = nodes.check_tool_needed_node(state, runtime=None)
    assert updated["need_tool"] is False
    assert updated.get("selected_tool", "") == ""


def test_expired_pending_tool_should_be_cleared():
    nodes = RAGNodes(
        llm=None,
        tools=[
            _DummyTool("hypertension_risk_assessment"),
            _DummyTool("diabetes_risk_assessment"),
        ],
    )
    state = {
        "messages": [HumanMessage(content="发烧了怎么办")],
        "pending_tool_name": "diabetes_risk_assessment",
        "pending_tool_deadline_ms": 1,
    }
    updated = nodes.check_tool_needed_node(state, runtime=None)
    assert updated.get("pending_tool_name", "") == ""
    assert updated.get("pending_tool_deadline_ms", 0) == 0
