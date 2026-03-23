import asyncio
import os
import sys
import time

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

    asyncio.run(_scenario())
    rag_api.eval_tasks.clear()


def test_prune_eval_tasks_should_remove_stale_finished_tasks():
    now_ts = time.time()
    rag_api.eval_tasks.clear()
    rag_api.eval_tasks["old"] = {
        "status": "completed",
        "finished_at": now_ts - rag_api.EVAL_TASK_RETENTION_SECONDS - 1,
    }
    rag_api.eval_tasks["fresh"] = {
        "status": "completed",
        "finished_at": now_ts,
    }

    rag_api._prune_eval_tasks(now_ts=now_ts)

    assert "old" not in rag_api.eval_tasks
    assert "fresh" in rag_api.eval_tasks
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
                {"question": "q1", "answer": "a1", "contexts_count": 2, "latency_ms": 10},
                {"question": "q2", "answer": "未在给定上下文中找到答案", "contexts_count": 0, "latency_ms": 12},
                {"question": "q3", "answer": "a3", "contexts_count": 1, "latency_ms": 8},
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
        assert saved_records, "应保存评测历史"

    asyncio.run(_scenario())
