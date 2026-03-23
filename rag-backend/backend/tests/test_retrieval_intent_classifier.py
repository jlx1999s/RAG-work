import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from backend.agent.graph.raggraph_node import RAGNodes
from backend.agent.models.raggraph_models import RetrievalMode
from backend.agent.models.retrieval_classifier import RetrievalIntentClassifier


def test_retrieval_intent_classifier_should_return_unavailable_when_model_missing():
    classifier = RetrievalIntentClassifier(
        model_path="/tmp/not-exists-retrieval-model.json",
        enabled=True,
    )
    result = classifier.predict("糖尿病治疗方案")
    assert result["available"] is False
    assert result["decision"] is None


def test_retrieval_intent_classifier_should_predict_probability_from_model(tmp_path: Path):
    model_path = tmp_path / "retrieval_intent_nb_v1.json"
    payload = {
        "model_version": "nb-test",
        "smoothing": 1.0,
        "vocab_size": 6,
        "class_counts": {"pos": 5, "neg": 5},
        "token_totals": {"pos": 20, "neg": 20},
        "token_counts": {
            "药": {"pos": 8, "neg": 0},
            "指": {"pos": 5, "neg": 1},
            "南": {"pos": 5, "neg": 1},
            "你": {"pos": 0, "neg": 7},
            "好": {"pos": 0, "neg": 7},
            "嗨": {"pos": 0, "neg": 5},
        },
    }
    model_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    classifier = RetrievalIntentClassifier(
        model_path=str(model_path),
        enabled=True,
        positive_threshold=0.7,
        negative_threshold=0.3,
    )

    pos_result = classifier.predict("药物指南")
    neg_result = classifier.predict("你好")
    uncertain_result = classifier.predict("今天")

    assert pos_result["available"] is True
    assert pos_result["decision"] is True
    assert neg_result["decision"] in {False, None}
    assert float(neg_result.get("probability") or 0.0) < 0.5
    assert uncertain_result["decision"] is None
    assert uncertain_result["band"] == "uncertain"


def test_check_retrieval_needed_should_use_statistical_classifier_before_llm():
    class _NoCallLLM:
        def with_structured_output(self, _schema):
            raise AssertionError("statistical classifier命中后不应继续调用LLM")

    class _StatClassifier:
        def predict(self, _question):
            return {
                "available": True,
                "decision": False,
                "reason": "statistical_classifier:certain_no",
                "model_version": "nb-test",
            }

    nodes = RAGNodes(llm=_NoCallLLM(), tools=[])
    nodes.retrieval_statistical_classifier = _StatClassifier()

    state = {
        "messages": [HumanMessage(content="请告诉我我是谁")],
        "retrieval_mode": RetrievalMode.AUTO,
    }
    updated = nodes.check_retrieval_needed_node(state, runtime=None)
    assert updated["need_retrieval"] is False
    stats = updated["retrieval_decision_stats"]
    assert stats["stage"] == "statistical_classifier"
    assert "fallback:statistical_classifier" in (stats.get("decision_path") or [])


def test_check_retrieval_needed_should_fallback_to_lightweight_when_statistical_uncertain():
    class _StructuredLLM:
        def with_structured_output(self, _schema):
            class _Invoker:
                def invoke(self, _prompt):
                    return SimpleNamespace(
                        need_retrieval=False,
                        extracted_question="测试问题",
                        reasoning="llm兜底"
                    )

            return _Invoker()

    class _StatClassifier:
        def predict(self, _question):
            return {
                "available": True,
                "decision": None,
                "reason": "uncertain",
                "model_version": "nb-test",
            }

    nodes = RAGNodes(llm=_StructuredLLM(), tools=[])
    nodes.retrieval_statistical_classifier = _StatClassifier()

    state = {
        "messages": [HumanMessage(content="谁在负责这个项目")],
        "retrieval_mode": RetrievalMode.AUTO,
    }
    updated = nodes.check_retrieval_needed_node(state, runtime=None)
    stats = updated["retrieval_decision_stats"]
    assert stats["stage"] in {"lightweight_classifier", "llm"}
    assert stats.get("statistical_classifier_result", {}).get("decision") is None


def test_check_retrieval_needed_should_skip_lightweight_by_default_when_statistical_uncertain():
    class _StructuredLLM:
        def with_structured_output(self, _schema):
            class _Invoker:
                def invoke(self, _prompt):
                    return SimpleNamespace(
                        need_retrieval=False,
                        extracted_question="测试问题",
                        reasoning="llm兜底"
                    )

            return _Invoker()

    class _StatClassifier:
        def predict(self, _question):
            return {
                "available": True,
                "decision": None,
                "reason": "uncertain",
                "model_version": "nb-test",
            }

    nodes = RAGNodes(llm=_StructuredLLM(), tools=[])
    nodes.retrieval_statistical_classifier = _StatClassifier()
    nodes.retrieval_lightweight_classifier_enabled = False

    state = {
        "messages": [HumanMessage(content="谁在负责这个项目")],
        "retrieval_mode": RetrievalMode.AUTO,
    }
    updated = nodes.check_retrieval_needed_node(state, runtime=None)
    stats = updated["retrieval_decision_stats"]
    assert stats["stage"] == "llm"
    assert "skip:lightweight_classifier" in (stats.get("decision_path") or [])


def test_check_retrieval_needed_should_use_lightweight_when_statistical_unavailable_and_enabled():
    class _StructuredLLM:
        def with_structured_output(self, _schema):
            raise AssertionError("lightweight命中后不应继续调用LLM")

    class _StatClassifier:
        def predict(self, _question):
            return {
                "available": False,
                "decision": None,
                "reason": "model_not_found",
                "model_version": "nb-test",
            }

    nodes = RAGNodes(llm=_StructuredLLM(), tools=[])
    nodes.retrieval_statistical_classifier = _StatClassifier()
    nodes.retrieval_lightweight_classifier_enabled = True
    nodes.retrieval_lightweight_classifier_only_when_stat_unavailable = True
    nodes.retrieval_classifier_yes_threshold = 1.0
    nodes.retrieval_classifier_margin = 1.0

    state = {
        "messages": [HumanMessage(content="谁在负责这个项目")],
        "retrieval_mode": RetrievalMode.AUTO,
    }
    updated = nodes.check_retrieval_needed_node(state, runtime=None)
    stats = updated["retrieval_decision_stats"]
    assert stats["stage"] == "lightweight_classifier"
