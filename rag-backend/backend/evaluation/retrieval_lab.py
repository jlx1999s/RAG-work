import math
import re
from typing import Any, Dict, List, Optional


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def _tokenize_text(text: str) -> set[str]:
    raw = _normalize_text(text).lower()
    if not raw:
        return set()
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", raw))


def _context_match_ratio(gold_context: str, context_text: str) -> float:
    gold_tokens = _tokenize_text(gold_context)
    ctx_tokens = _tokenize_text(context_text)
    if not gold_tokens or not ctx_tokens:
        return 0.0
    overlap = len(gold_tokens & ctx_tokens)
    return overlap / max(len(gold_tokens), 1)


def _doc_relevant(doc_text: str, gold_contexts: List[str], hit_threshold: float) -> bool:
    if not gold_contexts:
        return False
    best = 0.0
    for gold in gold_contexts:
        best = max(best, _context_match_ratio(gold, doc_text))
    return best >= hit_threshold


def _binary_ndcg_at_k(binary_rels: List[int], k: int) -> Optional[float]:
    if k <= 0:
        return None
    rels = binary_rels[:k]
    if not rels:
        return None
    dcg = 0.0
    for idx, rel in enumerate(rels, start=1):
        if rel <= 0:
            continue
        dcg += rel / math.log2(idx + 1)
    ideal = sorted(binary_rels, reverse=True)[:k]
    idcg = 0.0
    for idx, rel in enumerate(ideal, start=1):
        if rel <= 0:
            continue
        idcg += rel / math.log2(idx + 1)
    if idcg <= 0:
        return None
    return dcg / idcg


def _rank_metrics_for_docs(
    docs: List[str],
    gold_contexts: List[str],
    *,
    ks: List[int],
    hit_threshold: float,
) -> Dict[str, Any]:
    if not docs:
        return {
            "hit_at_k": {},
            "recall_at_k": {},
            "mrr": None,
            "ndcg_at_k": {},
            "relevant_docs": 0,
        }
    binary_rels = [1 if _doc_relevant(doc, gold_contexts, hit_threshold) else 0 for doc in docs]
    relevant_docs = sum(binary_rels)
    first_rel_rank = None
    for idx, rel in enumerate(binary_rels, start=1):
        if rel > 0:
            first_rel_rank = idx
            break
    mrr = (1.0 / first_rel_rank) if first_rel_rank else 0.0
    total_gold = max(1, len(gold_contexts))
    hit_at_k: Dict[str, Optional[float]] = {}
    recall_at_k: Dict[str, Optional[float]] = {}
    ndcg_at_k: Dict[str, Optional[float]] = {}
    for k in ks:
        rels_at_k = binary_rels[:k]
        hit_at_k[str(k)] = 1.0 if any(rels_at_k) else 0.0
        recall_at_k[str(k)] = min(1.0, sum(rels_at_k) / total_gold)
        ndcg_at_k[str(k)] = _binary_ndcg_at_k(binary_rels, k)
    return {
        "hit_at_k": hit_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "ndcg_at_k": ndcg_at_k,
        "relevant_docs": relevant_docs,
    }


def _aggregate_rank_metrics(
    rows: List[Dict[str, Any]],
    ks: List[int],
) -> Dict[str, Any]:
    if not rows:
        return {
            "hit_at_k": {str(k): None for k in ks},
            "recall_at_k": {str(k): None for k in ks},
            "ndcg_at_k": {str(k): None for k in ks},
            "mrr": None,
            "evaluated_items": 0,
        }
    hit_acc: Dict[str, List[float]] = {str(k): [] for k in ks}
    recall_acc: Dict[str, List[float]] = {str(k): [] for k in ks}
    ndcg_acc: Dict[str, List[float]] = {str(k): [] for k in ks}
    mrr_values: List[float] = []
    for row in rows:
        mrr = _safe_float(row.get("mrr"))
        if mrr is not None:
            mrr_values.append(mrr)
        for k in ks:
            k_key = str(k)
            hit_val = _safe_float((row.get("hit_at_k") or {}).get(k_key))
            if hit_val is not None:
                hit_acc[k_key].append(hit_val)
            recall_val = _safe_float((row.get("recall_at_k") or {}).get(k_key))
            if recall_val is not None:
                recall_acc[k_key].append(recall_val)
            ndcg_val = _safe_float((row.get("ndcg_at_k") or {}).get(k_key))
            if ndcg_val is not None:
                ndcg_acc[k_key].append(ndcg_val)
    return {
        "hit_at_k": {
            str(k): (sum(hit_acc[str(k)]) / len(hit_acc[str(k)])) if hit_acc[str(k)] else None
            for k in ks
        },
        "recall_at_k": {
            str(k): (sum(recall_acc[str(k)]) / len(recall_acc[str(k)])) if recall_acc[str(k)] else None
            for k in ks
        },
        "ndcg_at_k": {
            str(k): (sum(ndcg_acc[str(k)]) / len(ndcg_acc[str(k)])) if ndcg_acc[str(k)] else None
            for k in ks
        },
        "mrr": (sum(mrr_values) / len(mrr_values)) if mrr_values else None,
        "evaluated_items": len(rows),
    }


def _path_of_source(source: str) -> str:
    lower = _normalize_text(source).lower()
    if "graph" in lower:
        return "graph"
    if "vector" in lower:
        return "vector"
    return "other"


def _build_chunk_diagnostics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    chunk_lengths: List[float] = []
    total_chunks = 0
    matched_chunks = 0
    for item in items:
        gold_contexts = item.get("gold_contexts") or []
        details = item.get("context_details") or []
        if not isinstance(details, list):
            details = []
        for detail in details:
            if not isinstance(detail, dict):
                continue
            text = _normalize_text(detail.get("text"))
            if not text:
                continue
            total_chunks += 1
            chunk_lengths.append(float(len(text)))
            if _doc_relevant(text, gold_contexts, hit_threshold=0.3):
                matched_chunks += 1

    return {
        "chunk_hit_rate": _safe_divide(matched_chunks, total_chunks) if total_chunks > 0 else None,
        "chunk_length_distribution": {
            "count": len(chunk_lengths),
            "avg": (sum(chunk_lengths) / len(chunk_lengths)) if chunk_lengths else None,
            "p50": _percentile(chunk_lengths, 0.5),
            "p95": _percentile(chunk_lengths, 0.95),
            "max": max(chunk_lengths) if chunk_lengths else None,
            "min": min(chunk_lengths) if chunk_lengths else None,
        },
    }


def _build_rerank_diagnostics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged_docs_values: List[float] = []
    final_docs_values: List[float] = []
    for item in items:
        stats = item.get("retrieval_fusion_stats") or {}
        merged_docs = _safe_float(stats.get("merged_docs"))
        final_docs = _safe_float(stats.get("final_docs"))
        if merged_docs is not None:
            merged_docs_values.append(merged_docs)
        if final_docs is not None:
            final_docs_values.append(final_docs)

    merged_avg = (sum(merged_docs_values) / len(merged_docs_values)) if merged_docs_values else None
    final_avg = (sum(final_docs_values) / len(final_docs_values)) if final_docs_values else None
    pruning_rate = None
    if merged_avg is not None and final_avg is not None and merged_avg > 0:
        pruning_rate = 1.0 - (final_avg / merged_avg)
    return {
        "pre_rerank_avg_docs": merged_avg,
        "post_rerank_avg_docs": final_avg,
        "rerank_pruning_rate": pruning_rate,
        "rerank_uplift": None,
    }


def build_retrieval_lab_report(
    items: List[Dict[str, Any]],
    *,
    quality_summary: Dict[str, Any],
    hit_threshold: float = 0.3,
    ks: Optional[List[int]] = None,
) -> Dict[str, Any]:
    ks = ks or [1, 3, 5]
    ranked_rows_all: List[Dict[str, Any]] = []
    ranked_rows_vector: List[Dict[str, Any]] = []
    ranked_rows_graph: List[Dict[str, Any]] = []

    for item in items:
        gold_contexts = item.get("gold_contexts") or []
        if not gold_contexts:
            continue
        details = item.get("context_details") or []
        if not isinstance(details, list):
            details = []
        docs_all: List[str] = []
        docs_vector: List[str] = []
        docs_graph: List[str] = []
        for detail in details:
            if not isinstance(detail, dict):
                continue
            text = _normalize_text(detail.get("text"))
            if not text:
                continue
            docs_all.append(text)
            source = _normalize_text(detail.get("source"))
            path = _path_of_source(source)
            if path == "vector":
                docs_vector.append(text)
            elif path == "graph":
                docs_graph.append(text)
        if not docs_all:
            docs_all = [str(ctx) for ctx in (item.get("contexts") or []) if _normalize_text(ctx)]
        ranked_rows_all.append(
            _rank_metrics_for_docs(docs_all, gold_contexts, ks=ks, hit_threshold=hit_threshold)
        )
        ranked_rows_vector.append(
            _rank_metrics_for_docs(docs_vector, gold_contexts, ks=ks, hit_threshold=hit_threshold)
        )
        ranked_rows_graph.append(
            _rank_metrics_for_docs(docs_graph, gold_contexts, ks=ks, hit_threshold=hit_threshold)
        )

    overall = _aggregate_rank_metrics(ranked_rows_all, ks)
    vector_path = _aggregate_rank_metrics(ranked_rows_vector, ks)
    graph_path = _aggregate_rank_metrics(ranked_rows_graph, ks)
    chunk_diag = _build_chunk_diagnostics(items)
    rerank_diag = _build_rerank_diagnostics(items)

    return {
        "name": "Retrieval Lab",
        "enabled": True,
        "metrics": {
            "recall_at_k": overall.get("recall_at_k"),
            "hit_at_k": overall.get("hit_at_k"),
            "mrr": overall.get("mrr"),
            "ndcg_at_k": overall.get("ndcg_at_k"),
            "context_precision": _safe_float(quality_summary.get("context_precision")),
            "context_recall": _safe_float(quality_summary.get("context_recall")),
        },
        "path_breakdown": {
            "vector": vector_path,
            "graph": graph_path,
            "fused": overall,
        },
        "diagnostics": {
            "chunk": chunk_diag,
            "rerank": rerank_diag,
        },
    }

