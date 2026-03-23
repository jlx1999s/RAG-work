import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from backend.api import rag as rag_api


class _DummyGraph:
    def invoke(self, input_data, context=None):
        return {
            "messages": [{"role": "assistant", "content": "回退答案"}],
            "retrieved_docs": [
                {
                    "page_content": "糖尿病并发症管理",
                    "metadata": {"source": "vector"},
                }
            ],
            "retrieval_fusion_stats": {"vector_docs": 1},
        }


async def _run_in_threadpool_immediate(func, *args, **kwargs):
    return func(*args, **kwargs)


def test_parse_dataset_jsonl_should_reject_non_object_row():
    try:
        rag_api._parse_dataset_jsonl('"just a string"')
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "JSON对象" in str(exc)


def test_parse_dataset_jsonl_should_require_question_or_query():
    try:
        rag_api._parse_dataset_jsonl('{"reference":"abc"}')
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "question或query" in str(exc)


def test_run_single_evaluation_should_fallback_to_message_answer_and_dict_contexts():
    result = rag_api._run_single_evaluation(_DummyGraph(), "测试问题", context=None)

    assert result["answer"] == "回退答案"
    assert result["contexts"] == ["糖尿病并发症管理"]
    assert result["retrieval_sources"] == {"vector": 1}
    assert result["retrieval_fusion_stats"]["vector_docs"] == 1
    assert "routing_trace" in result
    assert result["routing_trace"]["stage"] == ""


def test_fallback_metrics_should_handle_chinese_without_spaces():
    metrics = rag_api._fallback_metrics(
        [
            {
                "reference": "糖尿病",
                "contexts": ["糖尿病并发症管理"],
            }
        ]
    )

    assert metrics["context_recall"] is not None and metrics["context_recall"] > 0
    assert metrics["context_precision"] is not None and metrics["context_precision"] > 0


def test_evaluate_status_should_be_user_scoped(monkeypatch):
    rag_api.eval_tasks.clear()

    async def _fake_run_eval_task(task_id, payload, current_user):
        rag_api.eval_tasks[task_id].update(
            {
                "status": "completed",
                "result": {"ok": True},
                "finished_at": time.time(),
            }
        )

    monkeypatch.setattr(rag_api, "_run_eval_task", _fake_run_eval_task)

    async def _scenario():
        payload = rag_api.EvalRequest(dataset_jsonl='{"question":"q","reference":"a"}')
        created = await rag_api.evaluate_rag_async(payload, current_user=1)
        task_id = created.data["task_id"]
        await asyncio.sleep(0)

        forbidden = await rag_api.evaluate_rag_status(task_id, current_user=2)
        assert forbidden.status == 500

        mine = await rag_api.evaluate_rag_status(task_id, current_user=1)
        assert mine.status == 200
        assert mine.data["task_id"] == task_id
        assert isinstance(mine.data.get("progress_events"), list)
        assert len(mine.data.get("progress_events") or []) >= 1

    asyncio.run(_scenario())
    rag_api.eval_tasks.clear()


def test_prune_eval_tasks_should_remove_stale_finished_tasks():
    now_ts = time.time()
    rag_api.eval_tasks.clear()
    rag_api.eval_tasks["old"] = {
        "status": "completed",
        "finished_at": now_ts - rag_api.EVAL_TASK_RETENTION_SECONDS - 1,
    }
    rag_api.eval_tasks["old_canceled"] = {
        "status": "canceled",
        "finished_at": now_ts - rag_api.EVAL_TASK_RETENTION_SECONDS - 1,
    }
    rag_api.eval_tasks["fresh"] = {
        "status": "completed",
        "finished_at": now_ts,
    }

    rag_api._prune_eval_tasks(now_ts=now_ts)

    assert "old" not in rag_api.eval_tasks
    assert "old_canceled" not in rag_api.eval_tasks
    assert "fresh" in rag_api.eval_tasks
    rag_api.eval_tasks.clear()


def test_cancel_rag_evaluation_should_mark_task_as_canceled():
    rag_api.eval_tasks.clear()
    task_id = "cancel-task-1"
    rag_api.eval_tasks[task_id] = {
        "status": "running",
        "user_id": "1",
        "eval_scope": rag_api.EvalScope.RAG_FULL,
        "submitted_at": time.time() - 5,
        "started_at": time.time() - 4,
        "progress": {
            "stage": "running",
            "percent": 33.0,
            "processed": 2,
            "total": 6,
            "message": "执行中",
            "lab": rag_api.EvalScope.RAG_FULL,
        },
        "progress_events": [],
        "runner": None,
    }

    async def _scenario():
        resp = await rag_api.cancel_rag_evaluation(task_id, current_user=1)
        assert resp.status == 200
        assert resp.data["status"] == "canceled"
        status = await rag_api.evaluate_rag_status(task_id, current_user=1)
        assert status.status == 200
        assert status.data["status"] == "canceled"
        assert status.data["progress"]["stage"] == "canceled"
        assert isinstance(status.data.get("progress_events"), list)
        assert len(status.data.get("progress_events") or []) >= 1

    asyncio.run(_scenario())
    rag_api.eval_tasks.clear()


def test_evaluate_payload_should_reject_invalid_config_without_running_models():
    async def _scenario():
        payload = rag_api.EvalRequest(
            dataset_jsonl='{"question":"q","reference":"a"}',
            retrieval_mode="invalid_mode",
        )
        result = await rag_api._evaluate_payload(payload, current_user=1)
        assert "error" in result
        assert "retrieval_mode" in result["error"]

        payload2 = rag_api.EvalRequest(
            dataset_jsonl='{"question":"q","reference":"a"}',
            max_retrieval_docs=0,
        )
        result2 = await rag_api._evaluate_payload(payload2, current_user=1)
        assert "error" in result2
        assert "max_retrieval_docs" in result2["error"]

    asyncio.run(_scenario())


def test_resolve_reference_text_should_support_ground_truths_and_answers():
    row1 = {"ground_truths": ["答案A", "答案B"]}
    row2 = {"answers": ["答案C"]}
    row3 = {"reference": "答案D"}

    assert rag_api._resolve_reference_text(row1) == "答案A"
    assert rag_api._resolve_reference_text(row2) == "答案C"
    assert rag_api._resolve_reference_text(row3) == "答案D"


def test_evaluate_payload_should_sample_ragas_when_ragas_limit_set(monkeypatch):
    async def _scenario():
        monkeypatch.setattr(rag_api, "run_in_threadpool", _run_in_threadpool_immediate)
        monkeypatch.setattr(rag_api, "get_rag_graph_for_collection", lambda _cid: _DummyGraph())

        def _fake_ragas_eval(items):
            for item in items:
                item["metrics"] = {
                    "context_precision": 0.5,
                    "context_recall": 0.5,
                    "answer_relevancy": 0.5,
                    "faithfulness": 0.5,
                }
            return {
                "metrics": {
                    "context_precision": 0.5,
                    "context_recall": 0.5,
                    "answer_relevancy": 0.5,
                    "faithfulness": 0.5,
                },
                "items": items,
            }

        monkeypatch.setattr(rag_api, "_run_ragas_eval", _fake_ragas_eval)

        payload = rag_api.EvalRequest(
            dataset_jsonl="\n".join(
                [
                    '{"question":"q1","reference":"r1"}',
                    '{"question":"q2","reference":"r2"}',
                    '{"question":"q3","reference":"r3"}',
                ]
            ),
            collection_id="kb_test",
            enable_ragas=True,
            ragas_limit=1,
        )

        result = await rag_api._evaluate_payload(payload, current_user=1)
        assert "error" not in result
        assert "仅评测前1条样本" in (result.get("warning") or "")
        assert result["run"]["performance_summary"]["ragas_evaluated_rows"] == 1
        assert result["items"][0]["metrics"]["context_precision"] == 0.5
        assert result["items"][1]["metrics"]["context_precision"] is None

    asyncio.run(_scenario())


def test_train_classifier_payload_should_return_model_and_thresholds():
    dataset_jsonl = "\n".join(
        [
            '{"question":"请查文档中的复诊流程","need_retrieval":true}',
            '{"question":"慢病管理手册在哪一章","need_retrieval":true}',
            '{"question":"给我一个鼓励的话","need_retrieval":false}',
            '{"question":"写一句朋友圈文案","need_retrieval":false}',
            '{"question":"请给出高血压指南来源","need_retrieval":true}',
            '{"question":"帮我翻译这句话","need_retrieval":false}',
        ]
    )
    payload = rag_api.TrainClassifierRequest(
        dataset_jsonl=dataset_jsonl,
        model_name="pytest_retrieval_intent_nb",
        split_strategy="random",
        valid_ratio=0.2,
        test_ratio=0.2,
        min_token_freq=1,
    )
    result = rag_api._train_classifier_payload(payload, current_user=1)
    assert "error" not in result
    model_path = Path(result["model_path"])
    assert model_path.exists()
    assert 0 <= result["recommended_thresholds"]["positive"] <= 1
    assert 0 <= result["recommended_thresholds"]["negative"] <= 1
    assert result["split"]["train_size"] >= 1
    model_path.unlink(missing_ok=True)


def test_eval_dataset_entries_should_merge_user_and_builtin_and_prefer_user(monkeypatch, tmp_path):
    user_dir = tmp_path / "user_ds"
    builtin_dir = tmp_path / "builtin_ds"
    user_dir.mkdir(parents=True, exist_ok=True)
    builtin_dir.mkdir(parents=True, exist_ok=True)

    (user_dir / "only_user.jsonl").write_text('{"question":"q","reference":"a"}\n', encoding="utf-8")
    (builtin_dir / "only_builtin.jsonl").write_text('{"question":"q","reference":"a"}\n', encoding="utf-8")
    (user_dir / "dup.jsonl").write_text('{"question":"u","reference":"a"}\n', encoding="utf-8")
    (builtin_dir / "dup.jsonl").write_text('{"question":"b","reference":"a"}\n', encoding="utf-8")

    monkeypatch.setattr(
        rag_api,
        "_get_eval_dataset_search_dirs",
        lambda _current_user: [("user", user_dir), ("builtin", builtin_dir)],
    )

    entries = rag_api._list_eval_dataset_entries(current_user=1)
    names = [item["name"] for item in entries]
    assert "only_user.jsonl" in names
    assert "only_builtin.jsonl" in names
    assert names.count("dup.jsonl") == 1

    dup_entry = next(item for item in entries if item["name"] == "dup.jsonl")
    assert dup_entry["source"] == "user"
    assert dup_entry["read_only"] is False


def test_resolve_eval_dataset_path_should_search_builtin_when_user_missing(monkeypatch, tmp_path):
    user_dir = tmp_path / "user_ds"
    builtin_dir = tmp_path / "builtin_ds"
    user_dir.mkdir(parents=True, exist_ok=True)
    builtin_dir.mkdir(parents=True, exist_ok=True)

    (builtin_dir / "train_set.jsonl").write_text('{"question":"q","reference":"a"}\n', encoding="utf-8")

    monkeypatch.setattr(
        rag_api,
        "_get_eval_dataset_search_dirs",
        lambda _current_user: [("user", user_dir), ("builtin", builtin_dir)],
    )

    resolved = rag_api._resolve_eval_dataset_path("train_set.jsonl", current_user=1)
    assert resolved is not None
    assert resolved["source"] == "builtin"
    assert resolved["read_only"] is True

def test_get_eval_history_should_support_excluding_items(monkeypatch):
    history_row = {
        "id": "h1",
        "user_id": "1",
        "result": {
            "summary": {"context_precision": 0.5},
            "items": [{"question": "q1"}, {"question": "q2"}],
        },
    }
    monkeypatch.setattr(rag_api, "_load_eval_history", lambda: [history_row])

    async def _scenario():
        detail = await rag_api.get_eval_history("h1", include_items=False, current_user=1)
        assert detail.status == 200
        assert "items" not in (detail.data.get("result") or {})
        assert detail.data["result"]["items_count"] == 2

    asyncio.run(_scenario())


def test_get_eval_history_items_should_return_pagination(monkeypatch):
    history_row = {
        "id": "h2",
        "user_id": "1",
        "result": {
            "items": [
                {
                    "question": "q1",
                    "answer": "a1",
                    "contexts_count": 2,
                    "latency_ms": 10,
                    "classifier_label_eval": {"has_label": True, "is_correct": True, "confusion": "tp"},
                },
                {
                    "question": "q2",
                    "answer": "未在给定上下文中找到答案",
                    "contexts_count": 0,
                    "latency_ms": 12,
                    "classifier_label_eval": {"has_label": True, "is_correct": False, "confusion": "fn"},
                },
                {
                    "question": "q3",
                    "answer": "a3",
                    "contexts_count": 1,
                    "latency_ms": 8,
                    "classifier_label_eval": {"has_label": False},
                },
            ]
        },
    }
    monkeypatch.setattr(rag_api, "_load_eval_history", lambda: [history_row])

    async def _scenario():
        page1 = await rag_api.get_eval_history_items("h2", page=1, page_size=2, current_user=1)
        assert page1.status == 200
        assert page1.data["total_items"] == 3
        assert page1.data["total_pages"] == 2
        assert len(page1.data["items"]) == 2
        assert page1.data["stats"]["abstained_count"] == 1
        assert page1.data["stats"]["classifier_labeled_count"] == 2
        assert page1.data["stats"]["classifier_accuracy"] == 0.5
        assert page1.data["stats"]["classifier_fn"] == 1

    asyncio.run(_scenario())


def test_run_single_evaluation_should_include_sop_trace():
    class _GraphWithSOP:
        def invoke(self, input_data, context=None):
            return {
                "final_answer": "请立即线下就医",
                "retrieved_docs": [],
                "handoff_required": True,
                "handoff_reason": "命中红线",
                "triage_level": "emergency",
                "medical_red_flags": ["胸痛"],
                "structured_decision_valid": True,
                "structured_decision_error": "",
                "extracted_symptoms": [{"name": "胸痛"}],
                "extracted_vitals": {"heart_rate": 120},
                "intervention_plan": {"summary": "转人工"},
                "medical_structured_output": {"is_medical": True},
            }

    result = rag_api._run_single_evaluation(_GraphWithSOP(), "我胸痛", context=None)
    assert result["sop_trace"]["is_medical"] is True
    assert result["sop_trace"]["handoff_required"] is True
    assert result["sop_trace"]["triage_level"] == "emergency"
    assert result["sop_trace"]["red_flags"] == ["胸痛"]
    assert result["sop_trace"]["symptoms"] == ["胸痛"]


def test_evaluate_payload_should_expose_sop_summary_and_persist_items_when_hidden(monkeypatch):
    saved_records = []

    class _SOPDummyGraph:
        def invoke(self, input_data, context=None):
            question = input_data.get("messages", [{}])[-1].get("content", "")
            is_emergency = "胸痛" in question
            return {
                "final_answer": "建议尽快就医" if is_emergency else "建议继续观察",
                "retrieved_docs": [],
                "retrieval_fusion_stats": {},
                "handoff_required": is_emergency,
                "handoff_reason": "命中红线" if is_emergency else "",
                "triage_level": "emergency" if is_emergency else "routine",
                "medical_red_flags": ["胸痛"] if is_emergency else [],
                "structured_decision_valid": True,
                "structured_decision_error": "",
                "extracted_symptoms": [{"name": "胸痛"}] if is_emergency else [{"name": "咳嗽"}],
                "extracted_vitals": {},
                "intervention_plan": {"summary": "转人工" if is_emergency else "生活方式建议"},
                "medical_structured_output": {"is_medical": True},
            }

    async def _scenario():
        monkeypatch.setattr(rag_api, "run_in_threadpool", _run_in_threadpool_immediate)
        monkeypatch.setattr(rag_api, "get_rag_graph_for_collection", lambda _cid: _SOPDummyGraph())
        monkeypatch.setattr(rag_api, "_append_eval_history", lambda record: saved_records.append(record))
        monkeypatch.setattr(rag_api, "_run_ragas_eval", lambda items: {"metrics": {"context_precision": 0.1}, "items": items})

        payload = rag_api.EvalRequest(
            dataset_jsonl="\n".join(
                [
                    '{"question":"我胸痛，呼吸困难","reference":"应急处理","expected_handoff":true}',
                    '{"question":"轻微咳嗽如何处理","reference":"居家观察","expected_handoff":false}',
                ]
            ),
            collection_id="kb_test",
            include_item_details=False,
            enable_ragas=True,
            ragas_limit=0,
        )
        result = await rag_api._evaluate_payload(payload, current_user=1)
        assert "error" not in result
        assert result["item_details_included"] is False
        assert result["items"] == []
        assert result["items_count"] == 2
        assert result["sop_summary"]["handoff_required_count"] == 1
        assert result["sop_summary"]["expected_handoff_count"] == 2
        assert result["sop_summary"]["handoff_accuracy"] == 1.0
        assert "routing_summary" in result
        assert result["routing_summary"]["total_items"] == 2
        assert saved_records, "应保存评测历史"
        history_items = saved_records[0]["result"]["items"]
        assert len(history_items) == 2
        assert "contexts" not in history_items[0]
        assert history_items[0]["sop"]["handoff_required"] is True

    asyncio.run(_scenario())


def test_evaluate_payload_should_include_quality_gates_baseline_and_slices(monkeypatch):
    saved_records = []

    class _EvalBestPracticeGraph:
        def invoke(self, input_data, context=None):
            question = input_data.get("messages", [{}])[-1].get("content", "")
            if "胸痛" in question:
                return {
                    "final_answer": "建议立即急诊就医",
                    "retrieved_docs": [
                        {"page_content": "胸痛伴呼吸困难需要紧急就医", "metadata": {"source": "vector"}}
                    ],
                    "retrieval_fusion_stats": {"vector_docs": 1},
                    "handoff_required": True,
                    "handoff_reason": "命中急危重症红线",
                    "triage_level": "emergency",
                    "medical_red_flags": ["胸痛", "呼吸困难"],
                    "structured_decision_valid": True,
                    "medical_structured_output": {"is_medical": True},
                    "extracted_symptoms": [{"name": "胸痛"}],
                    "intervention_plan": {"summary": "转人工"}
                }
            return {
                "final_answer": "建议规律作息并复诊",
                "retrieved_docs": [
                    {"page_content": "轻微咳嗽可先观察，若加重需就医", "metadata": {"source": "vector"}}
                ],
                "retrieval_fusion_stats": {"vector_docs": 1},
                "handoff_required": False,
                "handoff_reason": "",
                "triage_level": "routine",
                "medical_red_flags": [],
                "structured_decision_valid": True,
                "medical_structured_output": {"is_medical": True},
                "extracted_symptoms": [{"name": "咳嗽"}],
                "intervention_plan": {"summary": "居家管理"}
            }

    async def _scenario():
        monkeypatch.setattr(rag_api, "run_in_threadpool", _run_in_threadpool_immediate)
        monkeypatch.setattr(rag_api, "get_rag_graph_for_collection", lambda _cid: _EvalBestPracticeGraph())
        monkeypatch.setattr(rag_api, "_append_eval_history", lambda record: saved_records.append(record))
        monkeypatch.setattr(
            rag_api,
            "_load_eval_history",
            lambda: [
                {
                    "id": "baseline-1",
                    "user_id": "1",
                    "created_at": 1000,
                    "dataset_name": "med_eval",
                    "collection_id": "kb_test",
                    "retrieval_mode": "vector_only",
                    "run_tag": "baseline",
                    "result": {
                        "run": {
                            "quality_summary": {
                                "context_precision": 0.2,
                                "context_recall": 0.2,
                                "faithfulness": 0.4,
                                "answer_relevancy": 0.4
                            },
                            "stability_summary": {"error_rate": 0.2, "abstained_answer_rate": 0.3},
                            "sop_summary": {"handoff_rate": 0.2, "handoff_accuracy": 0.5},
                            "performance_summary": {"p95_latency_ms": 3000, "avg_latency_ms": 2000},
                            "retrieval_summary": {"context_presence_rate": 0.5}
                        }
                    }
                }
            ],
        )

        def _fake_ragas_eval(items):
            for item in items:
                item["metrics"] = {
                    "context_precision": 0.8,
                    "context_recall": 0.8,
                    "answer_relevancy": 0.8,
                    "faithfulness": 0.8,
                }
            return {
                "metrics": {
                    "context_precision": 0.8,
                    "context_recall": 0.8,
                    "answer_relevancy": 0.8,
                    "faithfulness": 0.8,
                },
                "items": items,
            }

        monkeypatch.setattr(rag_api, "_run_ragas_eval", _fake_ragas_eval)

        payload = rag_api.EvalRequest(
            dataset_jsonl="\n".join(
                [
                    '{"question":"我胸痛且呼吸困难","reference":"应急处理","expected_handoff":true,"expected_contexts":["胸痛伴呼吸困难需要紧急就医"],"expected_sources":["vector"],"query_type":"triage","difficulty":"hard","risk_level":"high"}',
                    '{"question":"轻微咳嗽怎么办","reference":"居家观察","expected_handoff":false,"expected_contexts":["轻微咳嗽可先观察"],"expected_sources":["vector"],"query_type":"followup","difficulty":"easy","risk_level":"low"}',
                ]
            ),
            collection_id="kb_test",
            dataset_name="med_eval",
            retrieval_mode="vector_only",
            enable_ragas=True,
            ragas_limit=0,
        )
        result = await rag_api._evaluate_payload(payload, current_user=1)
        assert "error" not in result
        assert result["quality_gate_summary"]["overall_status"] in {"pass", "fail"}
        assert result["baseline_comparison"]["found"] is True
        assert "faithfulness" in result["baseline_comparison"]["deltas"]
        assert result["slice_summary"]["risk_level"]["high"]["count"] == 1
        assert result["retrieval_summary"]["labeled_context_count"] == 2
        assert result["retrieval_summary"]["labeled_source_count"] == 2
        assert result["items"][0]["retrieval_label_eval"]["gold_context_hit"] is True
        assert "badcase_summary" in result
        assert "reproducibility_summary" in result
        assert "module_labs" in result
        assert result["module_labs"]["classifier"]["enabled"] is True
        assert result["module_labs"]["retrieval"]["enabled"] is True
        assert result["module_labs"]["generation"]["enabled"] is True
        assert result["module_labs"]["medical_safety"]["enabled"] is True
        assert result["run"]["reproducibility_summary"]["replay_key"]
        assert saved_records, "应保存评测历史"

    asyncio.run(_scenario())


def test_evaluate_payload_should_include_routing_summary(monkeypatch):
    class _RoutingGraph:
        def invoke(self, input_data, context=None):
            return {
                "final_answer": "测试回答",
                "retrieved_docs": [],
                "retrieval_fusion_stats": {},
                "need_retrieval": False,
                "need_retrieval_reason": "stat classifier决定无需检索",
                "original_question": "我是谁",
                "retrieval_decision_stats": {
                    "stage": "statistical_classifier",
                    "decision_path": ["fallback:statistical_classifier"],
                    "statistical_classifier_result": {
                        "available": True,
                        "decision": False,
                        "probability": 0.1,
                        "confidence": 0.8,
                        "band": "certain_no",
                        "model_version": "nb-test",
                        "thresholds": {"positive": 0.75, "negative": 0.25},
                    },
                    "classifier_result": {"decision": None},
                    "llm_result": {"decision": None, "reason": ""},
                    "rule_result": {"decision": None, "hit_rule_id": ""},
                },
            }

    async def _scenario():
        monkeypatch.setattr(rag_api, "run_in_threadpool", _run_in_threadpool_immediate)
        monkeypatch.setattr(rag_api, "get_rag_graph_for_collection", lambda _cid: _RoutingGraph())
        payload = rag_api.EvalRequest(
            dataset_jsonl='{"question":"我是谁","reference":"无法判断","need_retrieval":false}',
            collection_id="kb_test",
            enable_ragas=False,
        )
        result = await rag_api._evaluate_payload(payload, current_user=1)
        assert "error" not in result
        assert result["routing_summary"]["total_items"] == 1
        assert result["routing_summary"]["stage_distribution"]["statistical_classifier"] == 1
        assert result["routing_summary"]["llm_fallback_rate"] == 0.0
        assert result["items"][0]["routing"]["stage"] == "statistical_classifier"
        assert result["items"][0]["routing"]["statistical_classifier"]["decision"] is False
        assert result["classifier_summary"]["labeled_items"] == 1
        assert result["classifier_summary"]["accuracy"] == 1.0
        assert result["eval_labs"]["classifier_validation"]["enabled"] is True

    asyncio.run(_scenario())


def test_evaluate_payload_should_support_classifier_only_scope(monkeypatch):
    saved_records = []

    class _FakeClassifier:
        def __init__(self, model_path, **kwargs):
            self.model_path = model_path
            self.ready = True
            self.error_message = ""
            self.model_version = "fake-nb-v1"

        def predict(self, text):
            need = "指南" in text
            return {
                "available": True,
                "decision": need,
                "probability": 0.9 if need else 0.1,
                "confidence": 0.8,
                "band": "certain_yes" if need else "certain_no",
                "reason": "fake classifier",
                "model_version": self.model_version,
                "thresholds": {"positive": 0.75, "negative": 0.25},
            }

    async def _scenario():
        monkeypatch.setattr(rag_api, "RetrievalIntentClassifier", _FakeClassifier)
        monkeypatch.setattr(rag_api, "run_in_threadpool", _run_in_threadpool_immediate)
        monkeypatch.setattr(rag_api, "_append_eval_history", lambda record: saved_records.append(record))
        payload = rag_api.EvalRequest(
            eval_scope="classifier_only",
            dataset_jsonl="\n".join(
                [
                    '{"question":"请给我糖尿病指南","need_retrieval":true}',
                    '{"question":"你好","need_retrieval":false}',
                ]
            ),
            include_item_details=True,
        )
        result = await rag_api._evaluate_payload(payload, current_user=1)
        assert "error" not in result
        assert result["run"]["config"]["eval_scope"] == "classifier_only"
        assert result["classifier_summary"]["labeled_items"] == 2
        assert result["classifier_summary"]["accuracy"] == 1.0
        assert result["classifier_quality_gate_summary"]["overall_status"] in {"pass", "fail", "skipped"}
        assert result["eval_labs"]["classifier_validation"]["enabled"] is True
        assert "badcase_summary" in result
        assert "reproducibility_summary" in result
        assert result["module_labs"]["classifier"]["enabled"] is True
        assert result["module_labs"]["retrieval"]["enabled"] is False
        assert result["module_labs"]["generation"]["enabled"] is False
        assert result["module_labs"]["medical_safety"]["enabled"] is False
        assert len(result["items"]) == 2
        assert saved_records, "应保存评测历史"

    asyncio.run(_scenario())


def test_evaluate_payload_should_support_retrieval_only_scope(monkeypatch):
    class _RetrievalOnlyGraph:
        def invoke(self, input_data, context=None):
            return {
                "final_answer": "检索结果",
                "retrieved_docs": [
                    {"page_content": "糖尿病管理指南", "metadata": {"source": "vector"}},
                    {"page_content": "糖尿病并发症图谱", "metadata": {"source": "graph"}},
                ],
                "retrieval_fusion_stats": {"vector_docs": 1, "graph_docs": 1, "merged_docs": 2, "final_docs": 2},
                "need_retrieval": True,
                "need_retrieval_reason": "命中检索规则",
                "retrieval_decision_stats": {
                    "stage": "rule",
                    "decision_path": ["hit:retrieval.force_yes.1"],
                    "rule_result": {"decision": True, "hit_rule_id": "retrieval.force_yes.1"},
                    "statistical_classifier_result": {},
                    "classifier_result": {},
                    "llm_result": {},
                },
            }

    async def _scenario():
        monkeypatch.setattr(rag_api, "run_in_threadpool", _run_in_threadpool_immediate)
        monkeypatch.setattr(rag_api, "get_rag_graph_for_collection", lambda _cid: _RetrievalOnlyGraph())
        payload = rag_api.EvalRequest(
            eval_scope="retrieval_only",
            dataset_jsonl='{"question":"糖尿病如何管理","reference":"糖尿病管理","gold_contexts":["糖尿病管理指南"]}',
            collection_id="kb_test",
            enable_ragas=False,
        )
        result = await rag_api._evaluate_payload(payload, current_user=1)
        assert "error" not in result
        assert result["run"]["config"]["eval_scope"] == "retrieval_only"
        assert result["module_labs"]["retrieval"]["enabled"] is True
        assert result["module_labs"]["generation"]["enabled"] is False
        assert result["module_labs"]["medical_safety"]["enabled"] is False

    asyncio.run(_scenario())


def test_get_eval_history_badcases_should_filter_by_tag(monkeypatch):
    history_row = {
        "id": "h3",
        "user_id": "1",
        "result": {
            "items": [
                {
                    "question": "q1",
                    "answer": "",
                    "status": "error",
                    "badcase": {"is_badcase": True, "tags": ["runtime_error"], "severity": "high", "primary_tag": "runtime_error"},
                },
                {
                    "question": "q2",
                    "answer": "未在给定上下文中找到答案",
                    "status": "success",
                    "badcase": {"is_badcase": True, "tags": ["abstained_answer"], "severity": "medium", "primary_tag": "abstained_answer"},
                },
                {
                    "question": "q3",
                    "answer": "ok",
                    "status": "success",
                    "badcase": {"is_badcase": False, "tags": [], "severity": "none", "primary_tag": None},
                },
            ]
        },
    }
    monkeypatch.setattr(rag_api, "_load_eval_history", lambda: [history_row])

    async def _scenario():
        response = await rag_api.get_eval_history_badcases("h3", tag="runtime_error", current_user=1)
        assert response.status == 200
        assert response.data["total_items"] == 1
        assert response.data["items"][0]["question"] == "q1"
        assert response.data["summary"]["badcase_count"] == 1

    asyncio.run(_scenario())
