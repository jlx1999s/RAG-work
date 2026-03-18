import asyncio
import json
import os
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from backend.agent.contexts.raggraph_context import RAGContext
from backend.agent.graph.raggraph import RAGGraph
from backend.agent.models.raggraph_models import RetrievalMode
from backend.config.agent import get_rag_graph_for_collection
from backend.config.dependencies import get_current_user
from backend.config.log import get_logger
from backend.config.models import initialize_models
from backend.param.common import Response

router = APIRouter(
    prefix="/rag",
    tags=["RAG"]
)

logger = get_logger(__name__)
eval_tasks: Dict[str, Dict[str, Any]] = {}
EVAL_TASK_RETENTION_SECONDS = int(os.getenv("RAG_EVAL_TASK_RETENTION_SECONDS", "3600"))

class EvalRequest(BaseModel):
    dataset_jsonl: str
    limit: Optional[int] = 0
    workspace: str = "eval_ws"
    retrieval_mode: str = RetrievalMode.VECTOR_ONLY
    max_retrieval_docs: int = 3
    collection_id: Optional[str] = None
    dataset_name: Optional[str] = None
    run_tag: Optional[str] = None
    enable_ragas: bool = True
    include_item_details: bool = True
    cache_enabled: Optional[bool] = None
    cache_namespace: Optional[str] = None
    ragas_limit: int = 10


class SaveEvalDatasetRequest(BaseModel):
    name: str
    content: str
    overwrite: bool = False


VALID_RETRIEVAL_MODES = {
    RetrievalMode.VECTOR_ONLY,
    RetrievalMode.GRAPH_ONLY,
    RetrievalMode.HYBRID,
    RetrievalMode.NO_RETRIEVAL,
    RetrievalMode.AUTO
}


def _sanitize_user_scope(user_id: Any) -> str:
    user_scope = str(user_id or "").strip() or "anonymous"
    return re.sub(r"[^0-9A-Za-z_\-]", "_", user_scope)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _first_non_empty_text(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            text = _normalize_text(item)
            if text:
                return text
        return ""
    return _normalize_text(value)


def _resolve_reference_text(row: Dict[str, Any]) -> str:
    return (
        _first_non_empty_text(row.get("reference"))
        or _first_non_empty_text(row.get("answer"))
        or _first_non_empty_text(row.get("ground_truths"))
        or _first_non_empty_text(row.get("answers"))
    )


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                block_text = block.get("text")
                if block_text:
                    parts.append(str(block_text))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _extract_answer_from_messages(messages: Any) -> str:
    if not isinstance(messages, list) or not messages:
        return ""
    for message in reversed(messages):
        if message is None:
            continue
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        text = _message_content_to_text(content)
        if text:
            return text
    return ""


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, float) and value != value:
            return None
        return float(value)
    except Exception:
        return None


def _safe_divide(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _percentile(values: List[float], ratio: float) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = int(round((len(sorted_values) - 1) * ratio))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return float(sorted_values[idx])


def _aggregate_fusion_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    summary: Dict[str, Any] = {}
    candidate_keys = {
        "vector_docs",
        "graph_docs",
        "merged_docs",
        "rrf_k",
        "mmr_lambda",
        "selected_text_docs",
        "selected_chart_docs",
        "final_docs"
    }
    for key in candidate_keys:
        values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            summary[key] = round(sum(values) / len(values), 6)
    return summary


def _build_retrieval_summary(items: List[Dict[str, Any]], fusion_stats_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(items)
    if total == 0:
        return {"total_items": 0}
    contexts_counts = [int(item.get("contexts_count") or 0) for item in items]
    context_char_lengths = [int(item.get("contexts_total_chars") or 0) for item in items]
    no_context_count = sum(1 for count in contexts_counts if count == 0)
    with_context_count = total - no_context_count
    source_counter: Dict[str, int] = {}
    for item in items:
        for source_name, source_count in (item.get("retrieval_sources") or {}).items():
            if isinstance(source_count, int):
                source_counter[source_name] = source_counter.get(source_name, 0) + source_count
    return {
        "total_items": total,
        "with_context_count": with_context_count,
        "no_context_count": no_context_count,
        "context_presence_rate": _safe_float(_safe_divide(with_context_count, total)),
        "avg_contexts_per_item": round(sum(contexts_counts) / total, 4),
        "avg_context_chars_per_item": round(sum(context_char_lengths) / total, 2),
        "p95_contexts_per_item": _percentile([float(v) for v in contexts_counts], 0.95),
        "source_distribution": source_counter,
        "fusion_summary": _aggregate_fusion_stats(fusion_stats_rows)
    }


def _build_performance_summary(items: List[Dict[str, Any]], elapsed_ms: int) -> Dict[str, Any]:
    total = len(items)
    latencies = [float(item.get("latency_ms")) for item in items if isinstance(item.get("latency_ms"), (int, float))]
    if total == 0:
        return {"total_items": 0, "run_elapsed_ms": elapsed_ms}
    throughput_qps = _safe_divide(total, elapsed_ms / 1000) if elapsed_ms > 0 else None
    return {
        "total_items": total,
        "run_elapsed_ms": elapsed_ms,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "p99_latency_ms": _percentile(latencies, 0.99),
        "max_latency_ms": max(latencies) if latencies else None,
        "min_latency_ms": min(latencies) if latencies else None,
        "throughput_qps": round(throughput_qps, 4) if throughput_qps is not None else None
    }


def _build_stability_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(items)
    if total == 0:
        return {"total_items": 0}
    failed = sum(1 for item in items if item.get("status") == "error")
    empty_answer = sum(1 for item in items if not (item.get("answer") or "").strip())
    abstained_answer = sum(1 for item in items if _is_abstained_answer(item.get("answer")))
    return {
        "total_items": total,
        "failed_items": failed,
        "success_items": total - failed,
        "error_rate": _safe_float(_safe_divide(failed, total)),
        "empty_answer_count": empty_answer,
        "empty_answer_rate": _safe_float(_safe_divide(empty_answer, total)),
        "abstained_answer_count": abstained_answer,
        "abstained_answer_rate": _safe_float(_safe_divide(abstained_answer, total))
    }


def _parse_dataset_jsonl(raw_text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第{line_no}行JSON格式错误: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"第{line_no}行必须是JSON对象")
        question = _normalize_text(parsed.get("question") or parsed.get("query"))
        if not question:
            raise ValueError(f"第{line_no}行缺少question或query字段")
        rows.append(parsed)
    return rows


def _normalize_dataset_name(raw_name: str) -> str:
    name = (raw_name or "").strip()
    if not name:
        raise ValueError("评测数据集名称不能为空")
    if "/" in name or "\\" in name or ".." in name:
        raise ValueError("评测数据集名称不合法")
    if not re.match(r"^[\w\-\.\u4e00-\u9fff]+$", name):
        raise ValueError("评测数据集名称仅支持中英文、数字、下划线、短横线与点号")
    if not name.lower().endswith(".jsonl"):
        name = f"{name}.jsonl"
    return name


def _run_single_evaluation(
    rag_graph: RAGGraph,
    question: str,
    context: RAGContext
) -> Dict[str, Any]:
    start_time = time.time()
    input_data = {"messages": [{"role": "user", "content": question}]}
    output = rag_graph.invoke(input_data, context=context)
    answer = ""
    contexts: List[Any] = []
    fusion_stats: Dict[str, Any] = {}
    if isinstance(output, dict):
        answer = _normalize_text(output.get("final_answer"))
        if not answer:
            answer = _extract_answer_from_messages(output.get("messages") or [])
        contexts = output.get("retrieved_docs") or []
        fusion_stats = output.get("retrieval_fusion_stats") or {}
    else:
        answer = _normalize_text(output)
    normalized_contexts = []
    retrieval_sources: Dict[str, int] = {}
    contexts_total_chars = 0
    for ctx in contexts:
        if ctx is None:
            continue
        if isinstance(ctx, dict):
            metadata = ctx.get("metadata") or {}
            page_content = ctx.get("page_content")
        else:
            metadata = getattr(ctx, "metadata", {}) or {}
            page_content = getattr(ctx, "page_content", None)
        source_name = str(metadata.get("source") or "unknown")
        retrieval_sources[source_name] = retrieval_sources.get(source_name, 0) + 1
        if page_content is not None:
            content_text = str(page_content)
            normalized_contexts.append(content_text)
            contexts_total_chars += len(content_text)
        else:
            content_text = str(ctx)
            normalized_contexts.append(content_text)
            contexts_total_chars += len(content_text)
    elapsed_ms = int((time.time() - start_time) * 1000)
    return {
        "answer": answer,
        "contexts": normalized_contexts,
        "retrieval_fusion_stats": fusion_stats,
        "retrieval_sources": retrieval_sources,
        "contexts_total_chars": contexts_total_chars,
        "latency_ms": elapsed_ms
    }


def _tokenize_for_overlap(text: str) -> set[str]:
    raw_text = _normalize_text(text).lower()
    if not raw_text:
        return set()
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", raw_text))


def _is_abstained_answer(answer: Any) -> bool:
    text = _normalize_text(answer).lower()
    if not text:
        return True
    markers = [
        "未在给定上下文中找到答案",
        "无法根据提供的上下文",
        "根据提供的上下文无法",
        "not found in the provided context",
        "cannot answer from the provided context"
    ]
    return any(marker.lower() in text for marker in markers)


def _fallback_metrics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {"context_precision": None, "context_recall": None}
    precision_scores = []
    recall_scores = []
    for item in items:
        reference = _normalize_text(item.get("reference"))
        contexts = item.get("contexts") or []
        if not reference:
            precision_scores.append(None)
            recall_scores.append(None)
            continue
        reference_tokens = _tokenize_for_overlap(reference)
        if not reference_tokens:
            precision_scores.append(None)
            recall_scores.append(None)
            continue
        context_tokens = _tokenize_for_overlap(" ".join(str(ctx) for ctx in contexts)) if contexts else set()
        if not context_tokens:
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            continue
        overlap = reference_tokens & context_tokens
        precision_scores.append(len(overlap) / max(len(context_tokens), 1))
        recall_scores.append(len(overlap) / max(len(reference_tokens), 1))
    precision = [p for p in precision_scores if p is not None]
    recall = [r for r in recall_scores if r is not None]
    return {
        "context_precision": sum(precision) / len(precision) if precision else None,
        "context_recall": sum(recall) / len(recall) if recall else None
    }


def _run_ragas_eval(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {"metrics": {}, "items": []}
    eval_rows = []
    for item in items:
        eval_rows.append({
            "user_input": item.get("question") or "",
            "response": item.get("answer") or "",
            "retrieved_contexts": item.get("contexts") or [],
            "reference": item.get("reference") or ""
        })
    chat_model, embeddings_model = initialize_models()
    llm = LangchainLLMWrapper(chat_model)
    embeddings = LangchainEmbeddingsWrapper(embeddings_model)
    dataset = Dataset.from_list(eval_rows)
    metrics = [context_precision, context_recall, answer_relevancy, faithfulness]
    result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    df = result.to_pandas()
    metric_cols = [col for col in df.columns if col not in {"user_input", "response", "retrieved_contexts", "reference"}]
    summary = {col: _safe_float(df[col].mean()) for col in metric_cols}
    for idx, row in df.iterrows():
        items[idx]["metrics"] = {col: _safe_float(row[col]) for col in metric_cols}
    return {"metrics": summary, "items": items}


def _build_cost_summary(items: List[Dict[str, Any]], elapsed_ms: int) -> Dict[str, Any]:
    total = len(items)
    context_chars = sum(int(item.get("contexts_total_chars") or 0) for item in items)
    answer_chars = sum(len(item.get("answer") or "") for item in items)
    avg_context_chars = _safe_divide(context_chars, total) if total else None
    avg_answer_chars = _safe_divide(answer_chars, total) if total else None
    return {
        "estimated": {
            "total_context_chars": context_chars,
            "total_answer_chars": answer_chars,
            "avg_context_chars_per_item": round(avg_context_chars, 2) if avg_context_chars is not None else None,
            "avg_answer_chars_per_item": round(avg_answer_chars, 2) if avg_answer_chars is not None else None
        },
        "token_usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None
        },
        "unit_economics": {
            "avg_latency_ms_per_item": round(elapsed_ms / total, 2) if total else None
        }
    }


def _get_eval_dataset_dir(current_user: Any) -> Path:
    root = Path(__file__).resolve().parent.parent / "tests" / "eval_datasets"
    return root / f"user_{_sanitize_user_scope(current_user)}"


def _list_eval_dataset_files(current_user: Any) -> List[Path]:
    dataset_dir = _get_eval_dataset_dir(current_user)
    if not dataset_dir.exists():
        return []
    return sorted([p for p in dataset_dir.glob("*.jsonl") if p.is_file()])


def _get_eval_history_path() -> Path:
    return Path(__file__).resolve().parent.parent / "tests" / "eval_history.jsonl"


def _prune_eval_tasks(now_ts: Optional[float] = None) -> None:
    now = now_ts or time.time()
    stale_ids: List[str] = []
    for task_id, task in eval_tasks.items():
        status = task.get("status")
        finished_at = task.get("finished_at")
        if status not in {"completed", "failed"}:
            continue
        if not isinstance(finished_at, (int, float)):
            continue
        if now - float(finished_at) > EVAL_TASK_RETENTION_SECONDS:
            stale_ids.append(task_id)
    for task_id in stale_ids:
        eval_tasks.pop(task_id, None)


def _append_eval_history(record: Dict[str, Any]) -> None:
    history_path = _get_eval_history_path()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_eval_history() -> List[Dict[str, Any]]:
    history_path = _get_eval_history_path()
    if not history_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with history_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.append(json.loads(stripped))
                except Exception:
                    continue
    except Exception:
        return []
    return rows


def _find_eval_history_row(history_id: str, current_user: Any) -> Optional[Dict[str, Any]]:
    rows = _load_eval_history()
    for row in rows:
        if row.get("id") == history_id and str(row.get("user_id")) == str(current_user):
            return row
    return None


@router.get("/query")
async def query_rag(q: str):
    return {"query": q, "results": []}

@router.post("/index")
async def index_document(document: str):
    return {"status": "success", "document": document}


@router.post("/evaluate")
async def evaluate_rag(payload: EvalRequest, current_user: int = Depends(get_current_user)):
    response_payload = await _evaluate_payload(payload, current_user)
    if response_payload.get("error"):
        return Response.error(response_payload["error"])
    return Response.success(response_payload)


async def _evaluate_payload(payload: EvalRequest, current_user: int) -> Dict[str, Any]:
    start_time = time.time()
    if payload.limit is not None and payload.limit < 0:
        return {"error": "limit不能为负数"}
    if payload.max_retrieval_docs <= 0:
        return {"error": "max_retrieval_docs必须大于0"}
    if payload.ragas_limit < 0:
        return {"error": "ragas_limit不能为负数"}
    if payload.retrieval_mode not in VALID_RETRIEVAL_MODES:
        return {"error": f"不支持的retrieval_mode: {payload.retrieval_mode}"}
    try:
        rows = _parse_dataset_jsonl(payload.dataset_jsonl)
    except Exception as exc:
        return {"error": f"评测数据格式错误: {str(exc)}"}
    total_rows = len(rows)
    if payload.limit and payload.limit > 0:
        rows = rows[:payload.limit]
    if not rows:
        return {"error": "未提供有效的评测数据"}
    if payload.collection_id:
        rag_graph = get_rag_graph_for_collection(payload.collection_id)
    else:
        chat_model, embeddings_model = initialize_models()
        rag_graph = RAGGraph(
            llm=chat_model,
            embedding_model=embeddings_model,
            enable_checkpointer=False,
            workspace=payload.workspace
        )
    context = RAGContext(
        session_id=f"eval_{int(start_time)}",
        user_id=str(current_user),
        retrieval_mode=payload.retrieval_mode,
        max_retrieval_docs=payload.max_retrieval_docs
    )
    items: List[Dict[str, Any]] = []
    fusion_stats_rows: List[Dict[str, Any]] = []
    for row in rows:
        question = _normalize_text(row.get("question") or row.get("query"))
        reference = _resolve_reference_text(row)
        if not question:
            continue
        item_record: Dict[str, Any] = {
            "question": question,
            "reference": reference,
            "answer": "",
            "status": "success",
            "error": None,
            "latency_ms": None,
            "contexts": [],
            "contexts_count": 0,
            "contexts_total_chars": 0,
            "contexts_preview": [],
            "retrieval_sources": {},
            "retrieval_fusion_stats": {},
            "cache": {
                "enabled": payload.cache_enabled,
                "namespace": payload.cache_namespace,
                "hit": None
            }
        }
        try:
            eval_result = await run_in_threadpool(_run_single_evaluation, rag_graph, question, context)
            contexts = eval_result.get("contexts") or []
            fusion_stats = eval_result.get("retrieval_fusion_stats") or {}
            if fusion_stats:
                fusion_stats_rows.append(fusion_stats)
            item_record.update({
                "answer": eval_result.get("answer") or "",
                "latency_ms": eval_result.get("latency_ms"),
                "contexts": contexts,
                "contexts_count": len(contexts),
                "contexts_total_chars": int(eval_result.get("contexts_total_chars") or 0),
                "contexts_preview": [str(c)[:300] for c in contexts[:3]],
                "retrieval_sources": eval_result.get("retrieval_sources") or {},
                "retrieval_fusion_stats": fusion_stats
            })
        except Exception as exc:
            logger.error(f"单条评测失败: {exc}")
            item_record.update({
                "status": "error",
                "error": str(exc)
            })
        items.append(item_record)
    if not items:
        return {"error": "评测数据中未找到有效问题"}
    retrieval_elapsed_ms = int((time.time() - start_time) * 1000)
    metrics_result: Dict[str, Any] = {"metrics": {}, "items": items}
    ragas_elapsed_ms = 0
    ragas_evaluated_rows = 0
    warning_messages: List[str] = []
    if payload.enable_ragas:
        try:
            ragas_items = items
            if payload.ragas_limit > 0 and payload.ragas_limit < len(items):
                ragas_items = items[:payload.ragas_limit]
            ragas_start = time.time()
            metrics_result = await run_in_threadpool(_run_ragas_eval, ragas_items)
            ragas_elapsed_ms = int((time.time() - ragas_start) * 1000)
            ragas_evaluated_rows = len(ragas_items)
            if len(ragas_items) < len(items):
                for item in items[len(ragas_items):]:
                    item["metrics"] = {
                        "context_precision": None,
                        "context_recall": None,
                        "answer_relevancy": None,
                        "faithfulness": None
                    }
                metrics_result["items"] = items
                warning_messages.append(
                    f"RAGAS仅评测前{len(ragas_items)}条样本以控制耗时，可将ragas_limit设为0进行全量评测"
                )
        except Exception as exc:
            logger.error(f"RAGAS评测失败: {exc}")
            fallback = _fallback_metrics(items)
            for item in items:
                item["metrics"] = {
                    "context_precision": None,
                    "context_recall": None,
                    "answer_relevancy": None,
                    "faithfulness": None
                }
            metrics_result = {"metrics": fallback, "items": items}
            warning_messages.append("RAGAS评测失败，已返回基础指标")
    else:
        fallback = _fallback_metrics(items)
        for item in items:
            item["metrics"] = {
                "context_precision": None,
                "context_recall": None,
                "answer_relevancy": None,
                "faithfulness": None
            }
        metrics_result = {"metrics": fallback, "items": items}
        warning_messages.append("已关闭RAGAS，仅返回基础指标")
    elapsed_ms = int((time.time() - start_time) * 1000)
    evaluated_items = metrics_result.get("items", items)
    fusion_summary = _aggregate_fusion_stats(fusion_stats_rows)
    retrieval_summary = _build_retrieval_summary(evaluated_items, fusion_stats_rows)
    performance_summary = _build_performance_summary(evaluated_items, elapsed_ms)
    stability_summary = _build_stability_summary(evaluated_items)
    performance_summary.update({
        "retrieval_generation_elapsed_ms": retrieval_elapsed_ms,
        "ragas_elapsed_ms": ragas_elapsed_ms,
        "non_retrieval_overhead_ms": max(0, elapsed_ms - retrieval_elapsed_ms - ragas_elapsed_ms),
        "ragas_evaluated_rows": ragas_evaluated_rows
    })
    cost_summary = _build_cost_summary(evaluated_items, elapsed_ms)
    abstained_rate = stability_summary.get("abstained_answer_rate")
    if isinstance(abstained_rate, (int, float)) and abstained_rate >= 0.4:
        warning_messages.append("较多回答为“未在给定上下文中找到答案”，请检查评测数据集与所选知识库是否匹配")
    if metrics_result.get("warning"):
        warning_messages.append(str(metrics_result.get("warning")))
    warning_text = "；".join(dict.fromkeys([msg for msg in warning_messages if msg])) or None
    cache_summary = {
        "enabled": payload.cache_enabled,
        "namespace": payload.cache_namespace,
        "hit_count": None,
        "miss_count": None,
        "hit_rate": None
    }
    run_report = {
        "run_id": str(uuid4()),
        "run_tag": payload.run_tag,
        "created_at": time.time(),
        "dataset": {
            "name": (payload.dataset_name or "").strip() or None,
            "total_rows": total_rows,
            "used_rows": len(rows),
            "request_size_bytes": len(payload.dataset_jsonl.encode("utf-8")) if payload.dataset_jsonl else 0
        },
        "config": {
            "workspace": payload.workspace,
            "collection_id": payload.collection_id,
            "retrieval_mode": payload.retrieval_mode,
            "max_retrieval_docs": payload.max_retrieval_docs,
            "enable_ragas": payload.enable_ragas,
            "ragas_limit": payload.ragas_limit,
            "limit": payload.limit or 0
        },
        "quality_summary": metrics_result.get("metrics", {}),
        "retrieval_summary": retrieval_summary,
        "performance_summary": performance_summary,
        "stability_summary": stability_summary,
        "cost_summary": cost_summary,
        "cache_summary": cache_summary
    }
    response_payload = {
        "run": run_report,
        "summary": metrics_result.get("metrics", {}),
        "retrieval_fusion_summary": fusion_summary,
        "retrieval_summary": retrieval_summary,
        "performance_summary": performance_summary,
        "stability_summary": stability_summary,
        "cost_summary": cost_summary,
        "cache_summary": cache_summary,
        "items": evaluated_items if payload.include_item_details else [],
        "elapsed_ms": elapsed_ms,
        "total": len(evaluated_items),
        "warning": warning_text
    }
    try:
        record = {
            "id": str(uuid4()),
            "created_at": time.time(),
            "user_id": str(current_user),
            "dataset_name": (payload.dataset_name or "").strip() or None,
            "dataset_total_lines": total_rows,
            "dataset_used_lines": len(rows),
            "dataset_size": len(payload.dataset_jsonl.encode("utf-8")) if payload.dataset_jsonl else 0,
            "workspace": payload.workspace,
            "retrieval_mode": payload.retrieval_mode,
            "max_retrieval_docs": payload.max_retrieval_docs,
            "collection_id": payload.collection_id,
            "enable_ragas": payload.enable_ragas,
            "run_tag": payload.run_tag,
            "cache_enabled": payload.cache_enabled,
            "cache_namespace": payload.cache_namespace,
            "limit": payload.limit or 0,
            "result": response_payload
        }
        _append_eval_history(record)
    except Exception as exc:
        logger.error(f"保存评测历史失败: {exc}")
    return response_payload


async def _run_eval_task(task_id: str, payload: EvalRequest, current_user: int) -> None:
    eval_tasks[task_id]["status"] = "running"
    try:
        result = await _evaluate_payload(payload, current_user)
        if result.get("error"):
            eval_tasks[task_id].update({
                "status": "failed",
                "error": result.get("error"),
                "finished_at": time.time()
            })
            return
        eval_tasks[task_id].update({
            "status": "completed",
            "result": result,
            "finished_at": time.time()
        })
    except Exception as exc:
        eval_tasks[task_id].update({
            "status": "failed",
            "error": str(exc),
            "finished_at": time.time()
        })


@router.post("/evaluate-async")
async def evaluate_rag_async(payload: EvalRequest, current_user: int = Depends(get_current_user)):
    _prune_eval_tasks()
    task_id = str(uuid4())
    eval_tasks[task_id] = {
        "status": "queued",
        "user_id": str(current_user),
        "submitted_at": time.time(),
        "result": None,
        "error": None
    }
    asyncio.create_task(_run_eval_task(task_id, payload, current_user))
    return Response.success({"task_id": task_id})


@router.get("/evaluate-status/{task_id}")
async def evaluate_rag_status(task_id: str, current_user: int = Depends(get_current_user)):
    _prune_eval_tasks()
    task = eval_tasks.get(task_id)
    if not task or str(task.get("user_id")) != str(current_user):
        return Response.error("评测任务不存在")
    payload = {
        "task_id": task_id,
        "status": task.get("status"),
        "submitted_at": task.get("submitted_at"),
        "finished_at": task.get("finished_at"),
        "error": task.get("error")
    }
    if task.get("status") == "completed":
        payload["result"] = task.get("result")
    return Response.success(payload)


@router.get("/eval-datasets")
async def list_eval_datasets(current_user: int = Depends(get_current_user)):
    items: List[Dict[str, Any]] = []
    for path in _list_eval_dataset_files(current_user):
        try:
            with path.open("r", encoding="utf-8") as f:
                line_count = sum(1 for line in f if line.strip())
        except Exception:
            line_count = 0
        size = path.stat().st_size
        label = path.stem.replace("_", " ").strip()
        items.append({
            "name": path.name,
            "label": label,
            "lines": line_count,
            "size": size
        })
    return Response.success(items)


@router.post("/eval-datasets")
async def save_eval_dataset(payload: SaveEvalDatasetRequest, current_user: int = Depends(get_current_user)):
    try:
        dataset_name = _normalize_dataset_name(payload.name)
        rows = _parse_dataset_jsonl(payload.content or "")
        if not rows:
            return Response.error("评测数据内容为空或格式不正确")
    except Exception as exc:
        return Response.error(str(exc))
    dataset_dir = _get_eval_dataset_dir(current_user)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / dataset_name
    if path.exists() and not payload.overwrite:
        return Response.error("评测数据集已存在，请更换名称或允许覆盖")
    try:
        path.write_text(payload.content.strip() + "\n", encoding="utf-8")
    except Exception as exc:
        return Response.error(f"保存评测数据失败: {str(exc)}")
    size = path.stat().st_size
    return Response.success({
        "name": path.name,
        "lines": len(rows),
        "size": size
    })


@router.get("/eval-datasets/{dataset_name}")
async def get_eval_dataset(dataset_name: str, current_user: int = Depends(get_current_user)):
    if "/" in dataset_name or "\\" in dataset_name:
        return Response.error("评测数据集不存在")
    dataset_dir = _get_eval_dataset_dir(current_user)
    path = dataset_dir / dataset_name
    if not path.exists() or path.suffix.lower() != ".jsonl":
        return Response.error("评测数据集不存在")
    content = path.read_text(encoding="utf-8")
    return Response.success({
        "name": dataset_name,
        "content": content
    })


@router.get("/eval-history")
async def list_eval_history(limit: int = 50, current_user: int = Depends(get_current_user)):
    rows = _load_eval_history()
    filtered = [r for r in rows if str(r.get("user_id")) == str(current_user)]
    filtered.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    sliced = filtered[: max(1, min(limit, 200))]
    items: List[Dict[str, Any]] = []
    for row in sliced:
        record = dict(row)
        result = record.get("result") or {}
        if isinstance(result, dict) and "items" in result:
            result = dict(result)
            result.pop("items", None)
            record["result"] = result
        items.append(record)
    return Response.success(items)


@router.get("/eval-history/{history_id}")
async def get_eval_history(
    history_id: str,
    include_items: bool = True,
    current_user: int = Depends(get_current_user)
):
    row = _find_eval_history_row(history_id, current_user)
    if row:
        record = dict(row)
        if not include_items:
            result = dict(record.get("result") or {})
            items = result.get("items") or []
            if isinstance(items, list):
                result.pop("items", None)
                result["items_count"] = len(items)
            record["result"] = result
        return Response.success(record)
    return Response.error("评测历史不存在")


@router.get("/eval-history/{history_id}/items")
async def get_eval_history_items(
    history_id: str,
    page: int = 1,
    page_size: int = 10,
    current_user: int = Depends(get_current_user)
):
    page = max(1, page)
    page_size = max(1, min(page_size, 50))
    row = _find_eval_history_row(history_id, current_user)
    if not row:
        return Response.error("评测历史不存在")

    result = row.get("result") or {}
    all_items = result.get("items") or []
    if not isinstance(all_items, list):
        all_items = []

    total_items = len(all_items)
    total_pages = max(1, (total_items + page_size - 1) // page_size) if total_items > 0 else 1
    page = min(page, total_pages)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_items = all_items[start_idx:end_idx]

    latencies = [float(item.get("latency_ms")) for item in all_items if isinstance(item.get("latency_ms"), (int, float))]
    contexts_counts = [int(item.get("contexts_count") or 0) for item in all_items]
    error_count = sum(1 for item in all_items if item.get("status") == "error")
    abstained_count = sum(1 for item in all_items if _is_abstained_answer(item.get("answer")))

    return Response.success({
        "history_id": history_id,
        "page": page,
        "page_size": page_size,
        "total_items": total_items,
        "total_pages": total_pages,
        "items": page_items,
        "stats": {
            "error_count": error_count,
            "abstained_count": abstained_count,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "avg_contexts_count": round(sum(contexts_counts) / len(contexts_counts), 2) if contexts_counts else None
        }
    })
