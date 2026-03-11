import asyncio
import json
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

class EvalRequest(BaseModel):
    dataset_jsonl: str
    limit: Optional[int] = 0
    workspace: str = "eval_ws"
    retrieval_mode: str = RetrievalMode.VECTOR_ONLY
    max_retrieval_docs: int = 3
    collection_id: Optional[str] = None
    dataset_name: Optional[str] = None


class SaveEvalDatasetRequest(BaseModel):
    name: str
    content: str
    overwrite: bool = False


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, float) and value != value:
            return None
        return float(value)
    except Exception:
        return None


def _parse_dataset_jsonl(raw_text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
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
    input_data = {"messages": [{"role": "user", "content": question}]}
    output = rag_graph.invoke(input_data, context=context)
    answer = output.get("final_answer") or ""
    contexts = output.get("retrieved_docs") or []
    normalized_contexts = []
    for ctx in contexts:
        if ctx is None:
            continue
        page_content = getattr(ctx, "page_content", None)
        if page_content is not None:
            normalized_contexts.append(str(page_content))
        else:
            normalized_contexts.append(str(ctx))
    return {
        "answer": answer,
        "contexts": normalized_contexts,
        "retrieval_fusion_stats": output.get("retrieval_fusion_stats") or {}
    }


def _fallback_metrics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {"context_precision": None, "context_recall": None}
    precision_scores = []
    recall_scores = []
    for item in items:
        reference = item.get("reference") or ""
        contexts = item.get("contexts") or []
        if not reference:
            precision_scores.append(None)
            recall_scores.append(None)
            continue
        reference_tokens = set(reference.split())
        if not reference_tokens:
            precision_scores.append(None)
            recall_scores.append(None)
            continue
        context_tokens = set(" ".join(contexts).split()) if contexts else set()
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


def _get_eval_dataset_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "tests"


def _list_eval_dataset_files() -> List[Path]:
    dataset_dir = _get_eval_dataset_dir()
    if not dataset_dir.exists():
        return []
    return sorted([p for p in dataset_dir.glob("*.jsonl") if p.is_file()])


def _get_eval_history_path() -> Path:
    return _get_eval_dataset_dir() / "eval_history.jsonl"


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
    rows = _parse_dataset_jsonl(payload.dataset_jsonl)
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
        question = row.get("question") or row.get("query") or ""
        reference = row.get("reference") or row.get("answer") or ""
        if not question:
            continue
        eval_result = await run_in_threadpool(_run_single_evaluation, rag_graph, question, context)
        contexts = eval_result.get("contexts") or []
        fusion_stats = eval_result.get("retrieval_fusion_stats") or {}
        if fusion_stats:
            fusion_stats_rows.append(fusion_stats)
        items.append({
            "question": question,
            "reference": reference,
            "answer": eval_result.get("answer") or "",
            "contexts": contexts,
            "contexts_count": len(contexts),
            "contexts_preview": [str(c)[:300] for c in contexts[:3]],
            "retrieval_fusion_stats": fusion_stats
        })
    metrics_result: Dict[str, Any] = {}
    try:
        metrics_result = await run_in_threadpool(_run_ragas_eval, items)
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
        metrics_result = {"metrics": fallback, "items": items, "warning": "RAGAS评测失败，已返回基础指标"}
    elapsed_ms = int((time.time() - start_time) * 1000)
    fusion_summary = {}
    if fusion_stats_rows:
        for key in {"vector_docs", "graph_docs", "merged_docs", "rrf_k", "mmr_lambda"}:
            values = [row.get(key) for row in fusion_stats_rows if isinstance(row.get(key), (int, float))]
            if values:
                fusion_summary[key] = round(sum(values) / len(values), 6)
    response_payload = {
        "summary": metrics_result.get("metrics", {}),
        "retrieval_fusion_summary": fusion_summary,
        "items": metrics_result.get("items", items),
        "elapsed_ms": elapsed_ms,
        "total": len(items),
        "warning": metrics_result.get("warning")
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
    task_id = str(uuid4())
    eval_tasks[task_id] = {
        "status": "queued",
        "submitted_at": time.time(),
        "result": None,
        "error": None
    }
    asyncio.create_task(_run_eval_task(task_id, payload, current_user))
    return Response.success({"task_id": task_id})


@router.get("/evaluate-status/{task_id}")
async def evaluate_rag_status(task_id: str, current_user: int = Depends(get_current_user)):
    task = eval_tasks.get(task_id)
    if not task:
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
    for path in _list_eval_dataset_files():
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
    dataset_dir = _get_eval_dataset_dir()
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
    dataset_dir = _get_eval_dataset_dir()
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
async def get_eval_history(history_id: str, current_user: int = Depends(get_current_user)):
    rows = _load_eval_history()
    for row in rows:
        if row.get("id") == history_id and str(row.get("user_id")) == str(current_user):
            return Response.success(row)
    return Response.error("评测历史不存在")
