import os
import sys

from langchain_core.messages import HumanMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from backend.agent.graph.raggraph_node import RAGNodes
from backend.agent.models.raggraph_models import RetrievedDocument


class _NoCallLLM:
    def invoke(self, prompt):
        raise AssertionError("handoff分支不应继续调用LLM")


def test_route_medical_guard_should_return_handoff_when_required():
    nodes = RAGNodes(llm=None, tools=[])
    decision = nodes.route_medical_guard({"handoff_required": True})
    assert decision == "handoff"


def test_generate_intervention_plan_should_fallback_without_llm():
    nodes = RAGNodes(llm=None, tools=[])
    state = {
        "original_question": "我最近血压偏高，头晕，应该怎么管理？",
        "medical_structured_output": {
            "triage_level": "routine",
            "intervention_focus": ["血压管理", "饮食干预"],
            "symptoms": [{"name": "头晕"}],
            "red_flags": []
        },
        "retrieved_docs": [
            RetrievedDocument(
                page_content="高血压管理建议包括低盐饮食和规律监测血压。",
                metadata={"document_name": "高血压科普指南"}
            )
        ],
        "handoff_required": False
    }

    updated = nodes.generate_intervention_plan_node(state, runtime=None)
    plan = updated.get("intervention_plan", {})
    assert plan.get("summary")
    assert isinstance(plan.get("lifestyle_actions"), list)
    assert len(plan.get("lifestyle_actions", [])) > 0
    assert isinstance(plan.get("followup_actions"), list)


def test_generate_answer_should_short_circuit_when_handoff_required():
    nodes = RAGNodes(llm=_NoCallLLM(), tools=[])
    state = {
        "original_question": "我胸痛并且呼吸困难",
        "handoff_required": True,
        "medical_red_flags": ["胸痛", "呼吸困难"],
        "handoff_reason": "命中急危重症红线",
        "retrieved_docs": []
    }

    updated = nodes.generate_answer_node(state, runtime=None)
    answer = updated.get("final_answer", "")
    assert "转人工" in answer or "线下就医" in answer
    assert "免责声明" in answer


def test_direct_answer_should_short_circuit_when_handoff_required():
    nodes = RAGNodes(llm=_NoCallLLM(), tools=[])
    state = {
        "messages": [HumanMessage(content="突然胸痛，喘不过气")],
        "handoff_required": True,
        "medical_red_flags": ["胸痛", "呼吸困难"],
        "handoff_reason": "命中急危重症红线"
    }

    updated = nodes.direct_answer_node(state, runtime=None)
    answer = updated.get("final_answer", "")
    assert "转人工" in answer or "线下就医" in answer
    assert "免责声明" in answer
