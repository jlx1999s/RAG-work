import asyncio
import hashlib
import json
import math
import os
import random
import time
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from backend.agent.contexts.raggraph_context import RAGContext
from backend.agent.graph.raggraph import RAGGraph
from backend.agent.models.retrieval_classifier import RetrievalIntentClassifier, tokenize_text
from backend.agent.models.raggraph_models import RetrievalMode
from backend.config.agent import get_rag_graph_for_collection
from backend.config.dependencies import get_current_user
from backend.config.log import get_logger
from backend.config.models import initialize_models
from backend.evaluation import (
    build_badcase_summary,
    build_classifier_lab_report,
    build_classifier_quality_gate_summary,
    build_classifier_summary,
    build_eval_labs_summary,
    build_generation_lab_report,
    build_item_classifier_label_eval,
    build_medical_safety_lab_report,
    build_retrieval_lab_report,
    detect_badcase_tags,
    parse_need_retrieval_label,
    run_classifier_lab,
)
from backend.param.common import Response

router = APIRouter(
    prefix="/rag",
    tags=["RAG"]
)

logger = get_logger(__name__)
eval_tasks: Dict[str, Dict[str, Any]] = {}
EVAL_TASK_RETENTION_SECONDS = int(os.getenv("RAG_EVAL_TASK_RETENTION_SECONDS", "3600"))
EVAL_CONTEXT_OVERLAP_HIT_THRESHOLD = float(os.getenv("RAG_EVAL_CONTEXT_OVERLAP_HIT_THRESHOLD", "0.3"))
EVAL_TASK_PROGRESS_EVENTS_MAX = int(os.getenv("RAG_EVAL_TASK_PROGRESS_EVENTS_MAX", "300"))


class EvalScope:
    RAG_FULL = "rag_full"
    CLASSIFIER_ONLY = "classifier_only"
    RETRIEVAL_ONLY = "retrieval_only"
    GENERATION_ONLY = "generation_only"
    MEDICAL_SAFETY_ONLY = "medical_safety_only"

class EvalRequest(BaseModel):
    dataset_jsonl: str
    limit: Optional[int] = 0
    eval_scope: str = EvalScope.RAG_FULL
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
    ci_mode: bool = False
    fail_on_gate: bool = False
    baseline_history_id: Optional[str] = None
    classifier_model_path: Optional[str] = None
    classifier_positive_threshold: float = 0.75
    classifier_negative_threshold: float = 0.25


class SaveEvalDatasetRequest(BaseModel):
    name: str
    content: str
    overwrite: bool = False


class TrainClassifierRequest(BaseModel):
    dataset_jsonl: str
    eval_dataset_jsonl: Optional[str] = None
    model_name: Optional[str] = None
    model_version: str = "nb-v1"
    seed: int = 42
    split_strategy: str = "random"
    valid_ratio: float = 0.2
    test_ratio: float = 0.1
    smoothing: float = 1.0
    min_token_freq: int = 2
    positive_threshold: Optional[float] = None
    negative_threshold: Optional[float] = None


VALID_RETRIEVAL_MODES = {
    RetrievalMode.VECTOR_ONLY,
    RetrievalMode.GRAPH_ONLY,
    RetrievalMode.HYBRID,
    RetrievalMode.NO_RETRIEVAL,
    RetrievalMode.AUTO
}

VALID_EVAL_SCOPES = {
    EvalScope.RAG_FULL,
    EvalScope.CLASSIFIER_ONLY,
    EvalScope.RETRIEVAL_ONLY,
    EvalScope.GENERATION_ONLY,
    EvalScope.MEDICAL_SAFETY_ONLY,
}

VALID_CLASSIFIER_SPLIT_STRATEGIES = {"random", "time"}


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


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _to_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "是"}:
        return True
    if text in {"false", "0", "no", "n", "否"}:
        return False
    return None


def _normalize_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        result: List[str] = []
        for item in value:
            text = _normalize_text(item)
            if text:
                result.append(text)
        return result
    text = _normalize_text(value)
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass
    if "||" in text:
        return [part.strip() for part in text.split("||") if part.strip()]
    return [text]


def _safe_divide(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _to_classifier_label(value: Any) -> Optional[int]:
    parsed = _to_optional_bool(value)
    if isinstance(parsed, bool):
        return 1 if parsed else 0
    text = _normalize_text(value).lower()
    if text in {"need", "需要"}:
        return 1
    if text in {"noneed", "不需要"}:
        return 0
    return None


def _to_timestamp(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _normalize_text(value)
    if not text:
        return None
    try:
        parsed = time.strptime(text.replace("T", " ").replace("Z", ""), "%Y-%m-%d %H:%M:%S")
        return float(time.mktime(parsed))
    except Exception:
        pass
    try:
        parsed = time.strptime(text, "%Y-%m-%d")
        return float(time.mktime(parsed))
    except Exception:
        return None


def _extract_classifier_timestamp(row: Dict[str, Any]) -> Optional[float]:
    for key in ("timestamp", "ts", "created_at", "createdAt", "date", "time"):
        if key not in row:
            continue
        ts = _to_timestamp(row.get(key))
        if ts is not None:
            return ts
    return None


def _parse_classifier_train_dataset_jsonl(raw_text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate((raw_text or "").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第{line_no}行JSON格式错误: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"第{line_no}行必须是JSON对象")
        text = _normalize_text(payload.get("question") or payload.get("query") or payload.get("text"))
        if not text:
            raise ValueError(f"第{line_no}行缺少question/query/text字段")
        label_value = payload.get("need_retrieval")
        if label_value is None and "label" in payload:
            label_value = payload.get("label")
        label = _to_classifier_label(label_value)
        if label is None:
            raise ValueError(f"第{line_no}行缺少need_retrieval/label或值不合法")
        rows.append(
            {
                "text": text,
                "label": int(label),
                "timestamp": _extract_classifier_timestamp(payload),
                "raw": payload,
            }
        )
    if not rows:
        raise ValueError("训练数据为空")
    return rows


def _split_classifier_samples(
    rows: List[Dict[str, Any]],
    *,
    seed: int,
    valid_ratio: float,
    test_ratio: float,
    strategy: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    prepared = list(rows)
    mode = strategy if strategy in VALID_CLASSIFIER_SPLIT_STRATEGIES else "random"
    if mode == "time":
        with_ts = [row for row in prepared if row.get("timestamp") is not None]
        without_ts = [row for row in prepared if row.get("timestamp") is None]
        if len(with_ts) >= max(3, int(len(prepared) * 0.6)):
            with_ts.sort(key=lambda x: float(x["timestamp"]))
            prepared = with_ts + without_ts
        else:
            mode = "random"
    if mode == "random":
        rng = random.Random(seed)
        rng.shuffle(prepared)

    total = len(prepared)
    test_size = int(total * max(0.0, test_ratio))
    valid_size = int(total * max(0.0, valid_ratio))
    if test_size <= 0 and total >= 10 and test_ratio > 0:
        test_size = 1
    if valid_size <= 0 and total >= 5 and valid_ratio > 0:
        valid_size = 1
    if test_size + valid_size >= total:
        overflow = test_size + valid_size - (total - 1)
        if overflow > 0:
            if valid_size >= overflow:
                valid_size -= overflow
            else:
                overflow -= valid_size
                valid_size = 0
                test_size = max(0, test_size - overflow)

    test_rows = prepared[:test_size] if test_size > 0 else []
    valid_rows = prepared[test_size:test_size + valid_size] if valid_size > 0 else []
    train_rows = prepared[test_size + valid_size:]
    if not train_rows:
        train_rows = list(prepared)
        valid_rows = []
        test_rows = []
    return train_rows, valid_rows, test_rows


def _train_nb_classifier_model(
    samples: List[Dict[str, Any]],
    *,
    smoothing: float,
    min_token_freq: int,
) -> Dict[str, Any]:
    token_counts = defaultdict(lambda: {"pos": 0, "neg": 0})
    class_counts = {"pos": 0, "neg": 0}
    token_totals = {"pos": 0, "neg": 0}
    token_freq = Counter()
    tokenized_samples: List[Tuple[List[str], int]] = []
    for sample in samples:
        tokens = tokenize_text(sample["text"])
        label = int(sample["label"])
        tokenized_samples.append((tokens, label))
        token_freq.update(tokens)

    min_freq = max(1, int(min_token_freq))
    for tokens, label in tokenized_samples:
        class_name = "pos" if label == 1 else "neg"
        class_counts[class_name] += 1
        for token in tokens:
            if token_freq[token] < min_freq:
                continue
            token_counts[token][class_name] += 1
            token_totals[class_name] += 1

    return {
        "token_counts": token_counts,
        "class_counts": class_counts,
        "token_totals": token_totals,
        "vocab_size": max(1, len(token_counts)),
        "smoothing": max(1e-6, float(smoothing)),
    }


def _predict_nb_classifier_probability(model: Dict[str, Any], text: str) -> float:
    tokens = tokenize_text(text)
    token_counts = model["token_counts"]
    class_counts = model["class_counts"]
    token_totals = model["token_totals"]
    vocab_size = max(1, int(model["vocab_size"]))
    smoothing = max(1e-6, float(model["smoothing"]))
    total_examples = max(1, int(class_counts["pos"]) + int(class_counts["neg"]))

    def _log_prob(class_name: str) -> float:
        prior = math.log((int(class_counts[class_name]) + 1.0) / (total_examples + 2.0))
        denominator = int(token_totals[class_name]) + smoothing * vocab_size
        if denominator <= 0:
            denominator = 1.0
        score = prior
        for token in tokens:
            stats = token_counts.get(token) or {"pos": 0, "neg": 0}
            likelihood = (int(stats[class_name]) + smoothing) / denominator
            score += math.log(max(likelihood, 1e-12))
        return score

    log_pos = _log_prob("pos")
    log_neg = _log_prob("neg")
    logit = log_pos - log_neg
    try:
        return 1.0 / (1.0 + math.exp(-logit))
    except OverflowError:
        return 1.0 if logit > 0 else 0.0


def _binary_metrics_from_prob_rows(prob_rows: List[Tuple[float, int]]) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    for prob, label in prob_rows:
        pred = 1 if prob >= 0.5 else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    total = len(prob_rows)
    accuracy = ((tp + tn) / total) if total else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "total": total,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _compute_pr_auc_from_prob_rows(prob_rows: List[Tuple[float, int]]) -> Optional[float]:
    if not prob_rows:
        return None
    sorted_rows = sorted(prob_rows, key=lambda x: x[0], reverse=True)
    pos_total = sum(1 for _, label in sorted_rows if label == 1)
    if pos_total == 0:
        return None
    tp = fp = 0
    curve: List[Tuple[float, float]] = []
    for prob, label in sorted_rows:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / pos_total
        curve.append((recall, precision))
    curve.sort(key=lambda x: x[0])
    area = 0.0
    prev_r, prev_p = 0.0, 1.0
    for recall, precision in curve:
        area += (recall - prev_r) * ((precision + prev_p) / 2.0)
        prev_r, prev_p = recall, precision
    if prev_r < 1.0:
        area += (1.0 - prev_r) * prev_p
    return round(area, 6)


def _compute_ece_from_prob_rows(prob_rows: List[Tuple[float, int]], num_bins: int = 10) -> Optional[float]:
    if not prob_rows:
        return None
    bins = [[] for _ in range(num_bins)]
    for prob, label in prob_rows:
        idx = min(num_bins - 1, max(0, int(prob * num_bins)))
        bins[idx].append((prob, label))
    total = len(prob_rows)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(prob for prob, _ in bucket) / len(bucket)
        avg_acc = sum(label for _, label in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(avg_conf - avg_acc)
    return round(ece, 6)


def _build_calibration_curve(prob_rows: List[Tuple[float, int]], num_bins: int = 10) -> List[Dict[str, Any]]:
    bins = [[] for _ in range(num_bins)]
    for prob, label in prob_rows:
        idx = min(num_bins - 1, max(0, int(prob * num_bins)))
        bins[idx].append((prob, label))
    curve: List[Dict[str, Any]] = []
    for index, bucket in enumerate(bins):
        if not bucket:
            curve.append(
                {
                    "bin_index": index,
                    "prob_min": round(index / num_bins, 4),
                    "prob_max": round((index + 1) / num_bins, 4),
                    "count": 0,
                    "avg_confidence": None,
                    "empirical_positive_rate": None,
                }
            )
            continue
        curve.append(
            {
                "bin_index": index,
                "prob_min": round(index / num_bins, 4),
                "prob_max": round((index + 1) / num_bins, 4),
                "count": len(bucket),
                "avg_confidence": round(sum(prob for prob, _ in bucket) / len(bucket), 6),
                "empirical_positive_rate": round(sum(label for _, label in bucket) / len(bucket), 6),
            }
        )
    return curve


def _suggest_classifier_thresholds(prob_rows: List[Tuple[float, int]]) -> Dict[str, Any]:
    if not prob_rows:
        return {"positive_threshold": 0.75, "negative_threshold": 0.25}
    candidates = [i / 100.0 for i in range(60, 96, 2)]
    best_threshold = 0.75
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0
    for threshold in candidates:
        tp = fp = fn = 0
        for prob, label in prob_rows:
            pred = 1 if prob >= threshold else 0
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 1:
                fn += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    negative_threshold = max(0.05, min(0.45, best_threshold - 0.2))
    uncertain_count = sum(1 for prob, _ in prob_rows if negative_threshold < prob < best_threshold)
    return {
        "positive_threshold": round(best_threshold, 4),
        "negative_threshold": round(negative_threshold, 4),
        "expected_at_positive_threshold": {
            "f1": round(best_f1, 6),
            "precision": round(best_precision, 6),
            "recall": round(best_recall, 6),
        },
        "uncertain_rate_estimate": round(uncertain_count / len(prob_rows), 6) if prob_rows else None,
    }


def _evaluate_nb_classifier_model(model: Dict[str, Any], samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    start = time.perf_counter()
    prob_rows: List[Tuple[float, int]] = []
    for sample in samples:
        prob = _predict_nb_classifier_probability(model, sample["text"])
        prob_rows.append((prob, int(sample["label"])))
    elapsed_ms = (time.perf_counter() - start) * 1000
    metrics = _binary_metrics_from_prob_rows(prob_rows)
    avg_latency_ms = (elapsed_ms / len(samples)) if samples else None
    metrics["avg_infer_latency_ms"] = round(avg_latency_ms, 6) if avg_latency_ms is not None else None
    metrics["pr_auc"] = _compute_pr_auc_from_prob_rows(prob_rows) if prob_rows else None
    metrics["ece"] = _compute_ece_from_prob_rows(prob_rows, num_bins=10) if prob_rows else None
    metrics["recall_at_positive"] = metrics.get("recall")
    metrics["prob_rows"] = prob_rows
    return metrics


def _sanitize_classifier_model_name(raw_name: Optional[str]) -> str:
    name = _normalize_text(raw_name)
    if not name:
        return f"retrieval_intent_nb_{int(time.time())}.json"
    cleaned = re.sub(r"[^0-9A-Za-z_\-\.]+", "_", name).strip("_")
    if not cleaned:
        cleaned = f"retrieval_intent_nb_{int(time.time())}"
    if not cleaned.lower().endswith(".json"):
        cleaned = f"{cleaned}.json"
    return cleaned


def _get_classifier_model_dir(current_user: Any) -> Path:
    user_scope = _sanitize_user_scope(current_user)
    return (
        Path(__file__).resolve().parent.parent
        / "agent"
        / "models"
        / "artifacts"
        / "users"
        / user_scope
    )


def _train_classifier_payload(payload: TrainClassifierRequest, current_user: Any) -> Dict[str, Any]:
    rows = _parse_classifier_train_dataset_jsonl(payload.dataset_jsonl)
    train_rows, valid_rows, test_rows = _split_classifier_samples(
        rows,
        seed=int(payload.seed),
        valid_ratio=max(0.0, float(payload.valid_ratio)),
        test_ratio=max(0.0, float(payload.test_ratio)),
        strategy=payload.split_strategy,
    )
    model_data = _train_nb_classifier_model(
        train_rows,
        smoothing=float(payload.smoothing),
        min_token_freq=int(payload.min_token_freq),
    )
    train_metrics = _evaluate_nb_classifier_model(model_data, train_rows)
    valid_metrics = _evaluate_nb_classifier_model(model_data, valid_rows) if valid_rows else None
    test_metrics = _evaluate_nb_classifier_model(model_data, test_rows) if test_rows else None

    threshold_rows = (valid_metrics or train_metrics).get("prob_rows") or []
    threshold_suggestion = _suggest_classifier_thresholds(threshold_rows)
    calibration_curve = _build_calibration_curve(threshold_rows, num_bins=10)

    model_dir = _get_classifier_model_dir(current_user)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_name = _sanitize_classifier_model_name(payload.model_name)
    model_path = model_dir / model_name

    serializable_model = {
        "model_type": "naive_bayes_retrieval_intent",
        "model_version": _normalize_text(payload.model_version) or "nb-v1",
        "created_at": int(time.time()),
        "smoothing": float(payload.smoothing),
        "min_token_freq": int(max(1, payload.min_token_freq)),
        "vocab_size": int(model_data["vocab_size"]),
        "class_counts": model_data["class_counts"],
        "token_totals": model_data["token_totals"],
        "token_counts": model_data["token_counts"],
        "metadata": {
            "split_strategy": payload.split_strategy,
            "seed": payload.seed,
            "train_size": len(train_rows),
            "valid_size": len(valid_rows),
            "test_size": len(test_rows),
            "train_metrics": {k: v for k, v in train_metrics.items() if k != "prob_rows"},
            "valid_metrics": {k: v for k, v in (valid_metrics or {}).items() if k != "prob_rows"} if valid_metrics else None,
            "test_metrics": {k: v for k, v in (test_metrics or {}).items() if k != "prob_rows"} if test_metrics else None,
            "threshold_suggestion": threshold_suggestion,
            "calibration_curve": calibration_curve,
            "global_metrics": {
                "pr_auc": _compute_pr_auc_from_prob_rows(threshold_rows),
                "ece": _compute_ece_from_prob_rows(threshold_rows, num_bins=10),
            },
        },
    }
    model_path.write_text(json.dumps(serializable_model, ensure_ascii=False), encoding="utf-8")

    recommended_positive = float(payload.positive_threshold) if payload.positive_threshold is not None else float(
        threshold_suggestion.get("positive_threshold") or 0.75
    )
    recommended_negative = float(payload.negative_threshold) if payload.negative_threshold is not None else float(
        threshold_suggestion.get("negative_threshold") or 0.25
    )
    if recommended_positive < recommended_negative:
        recommended_positive, recommended_negative = recommended_negative, recommended_positive

    external_eval_report = None
    eval_dataset_rows = []
    if _normalize_text(payload.eval_dataset_jsonl):
        eval_dataset_rows = _parse_classifier_train_dataset_jsonl(payload.eval_dataset_jsonl or "")
    if eval_dataset_rows:
        classifier = RetrievalIntentClassifier(
            model_path=str(model_path),
            enabled=True,
            positive_threshold=recommended_positive,
            negative_threshold=recommended_negative,
        )
        if not classifier.ready:
            return {"error": f"模型训练成功但加载失败: {classifier.error_message or model_path}"}
        classifier_lab_result = run_classifier_lab(
            [{"question": row["text"], "need_retrieval": bool(row["label"])} for row in eval_dataset_rows],
            classifier,
        )
        summary = classifier_lab_result.get("summary") or {}
        gate = classifier_lab_result.get("quality_gate") or {}
        external_eval_report = {
            "summary": summary,
            "quality_gate": gate,
            "report": build_classifier_lab_report(
                classifier_summary=summary,
                classifier_quality_gate=gate,
            ),
            "samples": len(eval_dataset_rows),
        }

    response_payload = {
        "model_path": str(model_path),
        "model_name": model_name,
        "model_version": serializable_model.get("model_version"),
        "recommended_thresholds": {
            "positive": round(recommended_positive, 4),
            "negative": round(recommended_negative, 4),
        },
        "threshold_suggestion": threshold_suggestion,
        "split": {
            "strategy": payload.split_strategy,
            "seed": payload.seed,
            "train_size": len(train_rows),
            "valid_size": len(valid_rows),
            "test_size": len(test_rows),
        },
        "metrics": {
            "train": {k: v for k, v in train_metrics.items() if k != "prob_rows"},
            "valid": {k: v for k, v in (valid_metrics or {}).items() if k != "prob_rows"} if valid_metrics else None,
            "test": {k: v for k, v in (test_metrics or {}).items() if k != "prob_rows"} if test_metrics else None,
        },
        "calibration_curve": calibration_curve,
        "external_eval": external_eval_report,
    }
    return response_payload


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


def _evaluate_retrieval_labels(item: Dict[str, Any]) -> Dict[str, Any]:
    expected_contexts = _normalize_text_list(item.get("gold_contexts"))
    expected_sources = _normalize_text_list(item.get("gold_sources"))
    contexts = item.get("contexts") or []
    contexts_text = [str(ctx or "") for ctx in contexts if str(ctx or "").strip()]
    retrieval_sources = item.get("retrieval_sources") or {}
    actual_source_names = set(str(name).lower() for name in retrieval_sources.keys())

    metrics = {
        "has_gold_contexts": bool(expected_contexts),
        "has_gold_sources": bool(expected_sources),
        "gold_context_hit": None,
        "gold_source_hit": None,
        "gold_context_match_ratio": None,
        "gold_source_match_ratio": None
    }

    if expected_sources:
        expected_source_names = [name.lower() for name in expected_sources]
        hit_count = sum(1 for name in expected_source_names if name in actual_source_names)
        metrics["gold_source_match_ratio"] = _safe_float(_safe_divide(hit_count, len(expected_source_names)))
        metrics["gold_source_hit"] = hit_count > 0

    if expected_contexts:
        if not contexts_text:
            metrics["gold_context_hit"] = False
            metrics["gold_context_match_ratio"] = 0.0
        else:
            expected_hits = 0
            for expected_context in expected_contexts:
                expected_tokens = _tokenize_for_overlap(expected_context)
                if not expected_tokens:
                    continue
                best_ratio = 0.0
                for context_text in contexts_text:
                    context_tokens = _tokenize_for_overlap(context_text)
                    if not context_tokens:
                        continue
                    overlap = len(expected_tokens & context_tokens)
                    ratio = overlap / max(len(expected_tokens), 1)
                    if ratio > best_ratio:
                        best_ratio = ratio
                if best_ratio >= EVAL_CONTEXT_OVERLAP_HIT_THRESHOLD:
                    expected_hits += 1
            match_ratio = _safe_float(_safe_divide(expected_hits, len(expected_contexts)))
            metrics["gold_context_match_ratio"] = match_ratio
            metrics["gold_context_hit"] = bool(match_ratio and match_ratio > 0)

    return metrics


def _build_retrieval_summary(items: List[Dict[str, Any]], fusion_stats_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(items)
    if total == 0:
        return {"total_items": 0}
    contexts_counts = [int(item.get("contexts_count") or 0) for item in items]
    context_char_lengths = [int(item.get("contexts_total_chars") or 0) for item in items]
    no_context_count = sum(1 for count in contexts_counts if count == 0)
    with_context_count = total - no_context_count
    source_counter: Dict[str, int] = {}
    labeled_context_count = 0
    labeled_context_hit_count = 0
    labeled_source_count = 0
    labeled_source_hit_count = 0
    context_match_ratios: List[float] = []
    source_match_ratios: List[float] = []
    for item in items:
        for source_name, source_count in (item.get("retrieval_sources") or {}).items():
            if isinstance(source_count, int):
                source_counter[source_name] = source_counter.get(source_name, 0) + source_count
        label_eval = item.get("retrieval_label_eval") or {}
        if bool(label_eval.get("has_gold_contexts")):
            labeled_context_count += 1
            if bool(label_eval.get("gold_context_hit")):
                labeled_context_hit_count += 1
            ratio = _safe_float(label_eval.get("gold_context_match_ratio"))
            if ratio is not None:
                context_match_ratios.append(ratio)
        if bool(label_eval.get("has_gold_sources")):
            labeled_source_count += 1
            if bool(label_eval.get("gold_source_hit")):
                labeled_source_hit_count += 1
            ratio = _safe_float(label_eval.get("gold_source_match_ratio"))
            if ratio is not None:
                source_match_ratios.append(ratio)
    return {
        "total_items": total,
        "with_context_count": with_context_count,
        "no_context_count": no_context_count,
        "context_presence_rate": _safe_float(_safe_divide(with_context_count, total)),
        "avg_contexts_per_item": round(sum(contexts_counts) / total, 4),
        "avg_context_chars_per_item": round(sum(context_char_lengths) / total, 2),
        "p95_contexts_per_item": _percentile([float(v) for v in contexts_counts], 0.95),
        "source_distribution": source_counter,
        "fusion_summary": _aggregate_fusion_stats(fusion_stats_rows),
        "labeled_context_count": labeled_context_count,
        "labeled_context_hit_rate": _safe_float(_safe_divide(labeled_context_hit_count, labeled_context_count))
        if labeled_context_count > 0
        else None,
        "avg_gold_context_match_ratio": round(sum(context_match_ratios) / len(context_match_ratios), 4)
        if context_match_ratios
        else None,
        "labeled_source_count": labeled_source_count,
        "labeled_source_hit_rate": _safe_float(_safe_divide(labeled_source_hit_count, labeled_source_count))
        if labeled_source_count > 0
        else None,
        "avg_gold_source_match_ratio": round(sum(source_match_ratios) / len(source_match_ratios), 4)
        if source_match_ratios
        else None
    }


def _build_routing_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(items)
    if total == 0:
        return {"total_items": 0}

    stage_counter: Dict[str, int] = {}
    need_retrieval_count = 0
    no_retrieval_count = 0
    stat_available_count = 0
    stat_decided_count = 0
    stat_uncertain_count = 0
    stat_probabilities: List[float] = []

    for item in items:
        routing = item.get("routing") or {}
        stage = _normalize_text(routing.get("stage")) or "unknown"
        stage_counter[stage] = stage_counter.get(stage, 0) + 1

        if bool(routing.get("need_retrieval")):
            need_retrieval_count += 1
        else:
            no_retrieval_count += 1

        stat_info = routing.get("statistical_classifier") or {}
        if stat_info.get("available") is True:
            stat_available_count += 1
            if stat_info.get("decision") is None:
                stat_uncertain_count += 1
            else:
                stat_decided_count += 1
            probability = _safe_float(stat_info.get("probability"))
            if probability is not None:
                stat_probabilities.append(probability)

    llm_fallback_count = stage_counter.get("llm", 0)
    rule_count = stage_counter.get("rule", 0)
    stat_stage_count = stage_counter.get("statistical_classifier", 0)
    lightweight_stage_count = stage_counter.get("lightweight_classifier", 0)
    error_fallback_count = stage_counter.get("fallback_error", 0)

    return {
        "total_items": total,
        "need_retrieval_count": need_retrieval_count,
        "no_retrieval_count": no_retrieval_count,
        "need_retrieval_rate": _safe_float(_safe_divide(need_retrieval_count, total)),
        "no_retrieval_rate": _safe_float(_safe_divide(no_retrieval_count, total)),
        "stage_distribution": stage_counter,
        "rule_stage_rate": _safe_float(_safe_divide(rule_count, total)),
        "statistical_stage_rate": _safe_float(_safe_divide(stat_stage_count, total)),
        "lightweight_stage_rate": _safe_float(_safe_divide(lightweight_stage_count, total)),
        "llm_fallback_rate": _safe_float(_safe_divide(llm_fallback_count, total)),
        "fallback_error_rate": _safe_float(_safe_divide(error_fallback_count, total)),
        "statistical_classifier_available_count": stat_available_count,
        "statistical_classifier_decided_count": stat_decided_count,
        "statistical_classifier_uncertain_count": stat_uncertain_count,
        "statistical_classifier_decision_rate": _safe_float(_safe_divide(stat_decided_count, total)),
        "avg_statistical_probability": round(sum(stat_probabilities) / len(stat_probabilities), 4)
        if stat_probabilities
        else None,
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


def _build_sop_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(items)
    if total == 0:
        return {"total_items": 0}

    medical_count = 0
    handoff_count = 0
    structured_valid_count = 0
    red_flag_total = 0
    triage_distribution: Dict[str, int] = {}
    expected_handoff_total = 0
    handoff_correct_count = 0
    handoff_false_negative = 0
    handoff_false_positive = 0

    for item in items:
        sop = item.get("sop") or {}
        is_medical = bool(sop.get("is_medical"))
        if is_medical:
            medical_count += 1
        handoff_required = bool(sop.get("handoff_required"))
        if handoff_required:
            handoff_count += 1
        if bool(sop.get("structured_decision_valid")):
            structured_valid_count += 1
        red_flags = sop.get("red_flags") or []
        red_flag_total += len(red_flags) if isinstance(red_flags, list) else 0
        triage_level = _normalize_text(sop.get("triage_level")) or "unknown"
        triage_distribution[triage_level] = triage_distribution.get(triage_level, 0) + 1

        expected_handoff = item.get("expected_handoff")
        if isinstance(expected_handoff, bool):
            expected_handoff_total += 1
            correct = handoff_required == expected_handoff
            if correct:
                handoff_correct_count += 1
            elif expected_handoff:
                handoff_false_negative += 1
            else:
                handoff_false_positive += 1

    return {
        "total_items": total,
        "medical_item_count": medical_count,
        "medical_item_rate": _safe_float(_safe_divide(medical_count, total)),
        "handoff_required_count": handoff_count,
        "handoff_rate": _safe_float(_safe_divide(handoff_count, total)),
        "structured_decision_valid_count": structured_valid_count,
        "structured_decision_valid_rate": _safe_float(_safe_divide(structured_valid_count, total)),
        "avg_red_flags_per_item": round(red_flag_total / total, 4),
        "triage_distribution": triage_distribution,
        "expected_handoff_count": expected_handoff_total,
        "handoff_accuracy": _safe_float(_safe_divide(handoff_correct_count, expected_handoff_total))
        if expected_handoff_total > 0
        else None,
        "handoff_false_negative_count": handoff_false_negative,
        "handoff_false_positive_count": handoff_false_positive,
        "handoff_recall": _safe_float(
            _safe_divide(
                expected_handoff_total - handoff_false_negative,
                expected_handoff_total
            )
        )
        if expected_handoff_total > 0
        else None
    }


def _build_answer_overlap_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(items)
    if total == 0:
        return {"total_items": 0}
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    f1_scores: List[float] = []
    covered_count = 0
    for item in items:
        reference = _normalize_text(item.get("reference"))
        answer = _normalize_text(item.get("answer"))
        ref_tokens = _tokenize_for_overlap(reference)
        ans_tokens = _tokenize_for_overlap(answer)
        if not ref_tokens or not ans_tokens:
            continue
        overlap = len(ref_tokens & ans_tokens)
        precision = overlap / max(len(ans_tokens), 1)
        recall = overlap / max(len(ref_tokens), 1)
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        if recall > 0:
            covered_count += 1
    return {
        "total_items": total,
        "evaluated_items": len(f1_scores),
        "answer_overlap_coverage_rate": _safe_float(_safe_divide(covered_count, total)),
        "avg_answer_overlap_precision": round(sum(precision_scores) / len(precision_scores), 4) if precision_scores else None,
        "avg_answer_overlap_recall": round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else None,
        "avg_answer_overlap_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else None
    }


def _build_slice_summary(items: List[Dict[str, Any]], field_name: str) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        field_value = _normalize_text(item.get(field_name)) or "unknown"
        buckets.setdefault(field_value, []).append(item)
    summary: Dict[str, Any] = {}
    for bucket, bucket_items in buckets.items():
        total = len(bucket_items)
        if total == 0:
            continue
        failed = sum(1 for item in bucket_items if item.get("status") == "error")
        handoff = sum(1 for item in bucket_items if bool((item.get("sop") or {}).get("handoff_required")))
        latencies = [
            float(item.get("latency_ms"))
            for item in bucket_items
            if isinstance(item.get("latency_ms"), (int, float))
        ]
        summary[bucket] = {
            "count": total,
            "error_rate": _safe_float(_safe_divide(failed, total)),
            "handoff_rate": _safe_float(_safe_divide(handoff, total)),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None
        }
    return summary


def _extract_metric_for_delta(payload: Dict[str, Any], path: List[str]) -> Optional[float]:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return _safe_float(current)


def _build_baseline_comparison(
    current_user: Any,
    payload: EvalRequest,
    response_payload: Dict[str, Any]
) -> Dict[str, Any]:
    rows = _load_eval_history()
    user_rows = [row for row in rows if str(row.get("user_id")) == str(current_user)]
    if not user_rows:
        return {"found": False}
    forced_baseline_id = _normalize_text(payload.baseline_history_id)
    if forced_baseline_id:
        forced = None
        for row in user_rows:
            if _normalize_text(row.get("id")) == forced_baseline_id:
                forced = row
                break
        if forced is None:
            return {"found": False, "reason": "baseline_history_id_not_found", "requested_id": forced_baseline_id}
        candidates = [forced]
    else:
        candidates = []
    target_dataset = (payload.dataset_name or "").strip().lower()
    target_collection = (payload.collection_id or "").strip().lower()
    target_mode = (payload.retrieval_mode or "").strip().lower()
    target_scope = (payload.eval_scope or EvalScope.RAG_FULL).strip().lower()
    for row in user_rows:
        if forced_baseline_id:
            continue
        row_result = row.get("result") or {}
        row_dataset = _normalize_text(row.get("dataset_name")).lower()
        row_collection = _normalize_text(row.get("collection_id")).lower()
        row_mode = _normalize_text(row.get("retrieval_mode")).lower()
        row_scope = _normalize_text(
            row.get("eval_scope")
            or ((row_result.get("run") or {}).get("config") or {}).get("eval_scope")
            or EvalScope.RAG_FULL
        ).lower()
        if target_dataset and row_dataset and row_dataset != target_dataset:
            continue
        if target_collection and row_collection and row_collection != target_collection:
            continue
        if target_mode and row_mode and row_mode != target_mode:
            continue
        if target_scope and row_scope and row_scope != target_scope:
            continue
        if not isinstance(row_result, dict):
            continue
        candidates.append(row)
    if not candidates:
        return {"found": False}
    baseline_row = max(candidates, key=lambda r: float(r.get("created_at") or 0))
    baseline_result = baseline_row.get("result") or {}
    metric_paths = {
        "context_precision": ["run", "quality_summary", "context_precision"],
        "context_recall": ["run", "quality_summary", "context_recall"],
        "faithfulness": ["run", "quality_summary", "faithfulness"],
        "answer_relevancy": ["run", "quality_summary", "answer_relevancy"],
        "error_rate": ["run", "stability_summary", "error_rate"],
        "abstained_answer_rate": ["run", "stability_summary", "abstained_answer_rate"],
        "handoff_rate": ["run", "sop_summary", "handoff_rate"],
        "handoff_accuracy": ["run", "sop_summary", "handoff_accuracy"],
        "p95_latency_ms": ["run", "performance_summary", "p95_latency_ms"],
        "avg_latency_ms": ["run", "performance_summary", "avg_latency_ms"],
        "context_presence_rate": ["run", "retrieval_summary", "context_presence_rate"],
        "classifier_accuracy": ["run", "classifier_summary", "accuracy"],
        "classifier_f1": ["run", "classifier_summary", "f1"],
        "classifier_recall": ["run", "classifier_summary", "recall"],
        "classifier_uncertain_rate": ["run", "classifier_summary", "uncertain_rate"],
        "retrieval_hit_at_3": ["run", "module_labs", "retrieval", "metrics", "hit_at_k", "3"],
        "retrieval_mrr": ["run", "module_labs", "retrieval", "metrics", "mrr"],
        "generation_completeness": ["run", "module_labs", "generation", "metrics", "completeness"],
        "generation_citation_coverage": ["run", "module_labs", "generation", "metrics", "citation_coverage"],
        "medical_redline_recall": ["run", "module_labs", "medical_safety", "metrics", "redline_recall"],
        "medical_hallucination_rate": ["run", "module_labs", "medical_safety", "metrics", "hallucination_rate"],
    }
    deltas: Dict[str, Any] = {}
    for name, path in metric_paths.items():
        current_value = _extract_metric_for_delta(response_payload, path)
        baseline_value = _extract_metric_for_delta(baseline_result, path)
        if current_value is None or baseline_value is None:
            continue
        deltas[name] = {
            "current": current_value,
            "baseline": baseline_value,
            "delta": round(current_value - baseline_value, 6)
        }
    return {
        "found": True,
        "baseline_id": baseline_row.get("id"),
        "baseline_created_at": baseline_row.get("created_at"),
        "baseline_dataset_name": baseline_row.get("dataset_name"),
        "baseline_run_tag": baseline_row.get("run_tag"),
        "baseline_forced": bool(forced_baseline_id),
        "deltas": deltas
    }


def _build_quality_gate_summary(
    quality_summary: Dict[str, Any],
    retrieval_summary: Dict[str, Any],
    performance_summary: Dict[str, Any],
    stability_summary: Dict[str, Any],
    sop_summary: Dict[str, Any]
) -> Dict[str, Any]:
    rules = [
        {
            "key": "error_rate",
            "label": "错误率",
            "value": _safe_float(stability_summary.get("error_rate")),
            "threshold": _safe_float(os.getenv("RAG_EVAL_GATE_MAX_ERROR_RATE", "0.05")),
            "operator": "<="
        },
        {
            "key": "abstained_answer_rate",
            "label": "拒答率",
            "value": _safe_float(stability_summary.get("abstained_answer_rate")),
            "threshold": _safe_float(os.getenv("RAG_EVAL_GATE_MAX_ABSTAINED_RATE", "0.35")),
            "operator": "<="
        },
        {
            "key": "context_presence_rate",
            "label": "上下文覆盖率",
            "value": _safe_float(retrieval_summary.get("context_presence_rate")),
            "threshold": _safe_float(os.getenv("RAG_EVAL_GATE_MIN_CONTEXT_PRESENCE_RATE", "0.8")),
            "operator": ">="
        },
        {
            "key": "faithfulness",
            "label": "事实一致性",
            "value": _safe_float(quality_summary.get("faithfulness")),
            "threshold": _safe_float(os.getenv("RAG_EVAL_GATE_MIN_FAITHFULNESS", "0.65")),
            "operator": ">="
        },
        {
            "key": "p95_latency_ms",
            "label": "P95延迟(ms)",
            "value": _safe_float(performance_summary.get("p95_latency_ms")),
            "threshold": _safe_float(os.getenv("RAG_EVAL_GATE_MAX_P95_LATENCY_MS", "25000")),
            "operator": "<="
        },
        {
            "key": "handoff_recall",
            "label": "转人工召回率",
            "value": _safe_float(sop_summary.get("handoff_recall")),
            "threshold": _safe_float(os.getenv("RAG_EVAL_GATE_MIN_HANDOFF_RECALL", "0.95")),
            "operator": ">="
        }
    ]
    checks: List[Dict[str, Any]] = []
    fail_count = 0
    pass_count = 0
    skip_count = 0
    for rule in rules:
        value = rule.get("value")
        threshold = rule.get("threshold")
        operator = rule.get("operator")
        status = "skipped"
        passed = None
        if value is not None and threshold is not None:
            if operator == "<=":
                passed = value <= threshold
            else:
                passed = value >= threshold
            status = "pass" if passed else "fail"
        if status == "pass":
            pass_count += 1
        elif status == "fail":
            fail_count += 1
        else:
            skip_count += 1
        checks.append({
            "key": rule["key"],
            "label": rule["label"],
            "operator": operator,
            "value": value,
            "threshold": threshold,
            "status": status
        })
    overall = "pass" if fail_count == 0 else "fail"
    return {
        "overall_status": overall,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "skip_count": skip_count,
        "checks": checks
    }


def _build_release_decision(
    *,
    ci_mode: bool,
    fail_on_gate: bool,
    quality_gate_summary: Dict[str, Any],
    medical_safety_lab: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    quality_status = _normalize_text((quality_gate_summary or {}).get("overall_status")) or "unknown"
    medical_status = _normalize_text(((medical_safety_lab or {}).get("hard_gate") or {}).get("overall_status")) or "unknown"
    reasons: List[str] = []
    blocked = False
    if quality_status == "fail":
        reasons.append("quality_gate_failed")
        blocked = True
    if medical_status == "fail":
        reasons.append("medical_safety_hard_gate_failed")
        blocked = True
    effective_fail_on_gate = bool(fail_on_gate or ci_mode)
    if blocked and not effective_fail_on_gate:
        action = "warn_only"
    elif blocked and effective_fail_on_gate:
        action = "block"
    else:
        action = "allow"
    return {
        "ci_mode": bool(ci_mode),
        "fail_on_gate": bool(fail_on_gate),
        "effective_fail_on_gate": effective_fail_on_gate,
        "quality_gate_status": quality_status,
        "medical_hard_gate_status": medical_status,
        "blocked": bool(blocked and effective_fail_on_gate),
        "action": action,
        "reasons": reasons,
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
    sop_trace: Dict[str, Any] = {}
    routing_trace: Dict[str, Any] = {}
    if isinstance(output, dict):
        answer = _normalize_text(output.get("final_answer"))
        if not answer:
            answer = _extract_answer_from_messages(output.get("messages") or [])
        contexts = output.get("retrieved_docs") or []
        fusion_stats = output.get("retrieval_fusion_stats") or {}
        retrieval_decision_stats = output.get("retrieval_decision_stats") or {}
        decision_path = retrieval_decision_stats.get("decision_path") or []
        stat_classifier_result = retrieval_decision_stats.get("statistical_classifier_result") or {}
        lightweight_classifier_result = retrieval_decision_stats.get("classifier_result") or {}
        routing_trace = {
            "need_retrieval": bool(output.get("need_retrieval")),
            "reason": _normalize_text(output.get("need_retrieval_reason")),
            "stage": _normalize_text(retrieval_decision_stats.get("stage")),
            "original_question": _normalize_text(output.get("original_question")),
            "decision_path_tail": list(decision_path[-3:]) if isinstance(decision_path, list) else [],
            "rule_decision": {
                "decision": (retrieval_decision_stats.get("rule_result") or {}).get("decision"),
                "hit_rule_id": (retrieval_decision_stats.get("rule_result") or {}).get("hit_rule_id"),
            },
            "statistical_classifier": {
                "available": stat_classifier_result.get("available"),
                "decision": stat_classifier_result.get("decision"),
                "probability": stat_classifier_result.get("probability"),
                "confidence": stat_classifier_result.get("confidence"),
                "band": stat_classifier_result.get("band"),
                "model_version": stat_classifier_result.get("model_version"),
                "thresholds": stat_classifier_result.get("thresholds"),
            },
            "lightweight_classifier": {
                "decision": lightweight_classifier_result.get("decision"),
                "yes_score": lightweight_classifier_result.get("yes_score"),
                "no_score": lightweight_classifier_result.get("no_score"),
                "margin": lightweight_classifier_result.get("margin"),
            },
            "llm_decision": {
                "decision": (retrieval_decision_stats.get("llm_result") or {}).get("decision"),
                "reason": _normalize_text((retrieval_decision_stats.get("llm_result") or {}).get("reason")),
            },
        }
        medical_structured_output = output.get("medical_structured_output") or {}
        symptoms_raw = output.get("extracted_symptoms") or []
        symptom_names: List[str] = []
        if isinstance(symptoms_raw, list):
            for item in symptoms_raw:
                if isinstance(item, dict):
                    text = _normalize_text(item.get("name"))
                else:
                    text = _normalize_text(item)
                if text:
                    symptom_names.append(text)
        sop_trace = {
            "is_medical": bool(
                medical_structured_output.get("is_medical")
                if isinstance(medical_structured_output, dict)
                else False
            ),
            "handoff_required": bool(output.get("handoff_required")),
            "handoff_reason": _normalize_text(output.get("handoff_reason")),
            "triage_level": _normalize_text(output.get("triage_level")) or "unknown",
            "red_flags": [str(flag) for flag in (output.get("medical_red_flags") or []) if str(flag).strip()],
            "structured_decision_valid": bool(output.get("structured_decision_valid")),
            "structured_decision_error": _normalize_text(output.get("structured_decision_error")),
            "symptoms": symptom_names[:12],
            "vitals": output.get("extracted_vitals") or {},
            "intervention_plan": output.get("intervention_plan") or {}
        }
    else:
        answer = _normalize_text(output)
    normalized_contexts = []
    context_details: List[Dict[str, Any]] = []
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
        context_details.append({
            "source": source_name,
            "text": content_text,
            "char_len": len(content_text),
            "metadata": metadata if isinstance(metadata, dict) else {},
        })

    if not retrieval_sources and isinstance(output, dict):
        answer_sources = output.get("answer_sources") or []
        for source in answer_sources:
            if not isinstance(source, dict):
                continue
            source_name = str(source.get("retrieval_mode") or "unknown")
            retrieval_sources[source_name] = retrieval_sources.get(source_name, 0) + 1

    elapsed_ms = int((time.time() - start_time) * 1000)
    return {
        "answer": answer,
        "contexts": normalized_contexts,
        "retrieval_fusion_stats": fusion_stats,
        "retrieval_sources": retrieval_sources,
        "context_details": context_details,
        "contexts_total_chars": contexts_total_chars,
        "latency_ms": elapsed_ms,
        "sop_trace": sop_trace,
        "routing_trace": routing_trace,
    }


def _compact_items_for_history(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compacted: List[Dict[str, Any]] = []
    for item in items:
        compact = dict(item or {})
        contexts_preview = compact.get("contexts_preview")
        if not isinstance(contexts_preview, list) or not contexts_preview:
            contexts = compact.get("contexts") or []
            contexts_preview = [str(ctx)[:300] for ctx in contexts[:3]]
        compact["contexts_preview"] = contexts_preview
        context_details = compact.get("context_details")
        if isinstance(context_details, list):
            compact["context_details"] = [
                {
                    "source": _normalize_text((detail or {}).get("source")) or "unknown",
                    "char_len": _safe_int((detail or {}).get("char_len")) or len(_normalize_text((detail or {}).get("text"))),
                    "text_preview": _normalize_text((detail or {}).get("text"))[:180],
                }
                for detail in context_details[:8]
                if isinstance(detail, dict)
            ]
        compact.pop("contexts", None)
        routing = compact.get("routing")
        if isinstance(routing, dict):
            llm_decision = routing.get("llm_decision")
            if isinstance(llm_decision, dict):
                llm_reason = _normalize_text(llm_decision.get("reason"))
                if llm_reason and len(llm_reason) > 240:
                    llm_decision = dict(llm_decision)
                    llm_decision["reason"] = llm_reason[:240] + "..."
                    routing = dict(routing)
                    routing["llm_decision"] = llm_decision
            decision_path_tail = routing.get("decision_path_tail")
            if isinstance(decision_path_tail, list) and len(decision_path_tail) > 5:
                routing = dict(routing)
                routing["decision_path_tail"] = decision_path_tail[-5:]
            compact["routing"] = routing
        generation_alignment = compact.get("generation_alignment")
        if isinstance(generation_alignment, dict):
            alignments = generation_alignment.get("alignments") or []
            if isinstance(alignments, list):
                generation_alignment = dict(generation_alignment)
                generation_alignment["alignments"] = alignments[:6]
            compact["generation_alignment"] = generation_alignment
        compacted.append(compact)
    return compacted


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


def _get_builtin_eval_dataset_dirs() -> List[Path]:
    tests_root = Path(__file__).resolve().parent.parent / "tests"
    candidates = [
        tests_root / "data",
        tests_root / "eval_datasets" / "shared",
    ]
    return [path for path in candidates if path.exists() and path.is_dir()]


def _get_eval_dataset_search_dirs(current_user: Any) -> List[Tuple[str, Path]]:
    search_dirs: List[Tuple[str, Path]] = [("user", _get_eval_dataset_dir(current_user))]
    for builtin_dir in _get_builtin_eval_dataset_dirs():
        search_dirs.append(("builtin", builtin_dir))
    return search_dirs


def _list_eval_dataset_files(current_user: Any) -> List[Path]:
    dataset_dir = _get_eval_dataset_dir(current_user)
    if not dataset_dir.exists():
        return []
    return sorted([p for p in dataset_dir.glob("*.jsonl") if p.is_file()])


def _list_eval_dataset_entries(current_user: Any) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    seen_names: set[str] = set()
    for source, dataset_dir in _get_eval_dataset_search_dirs(current_user):
        if not dataset_dir.exists():
            continue
        for path in sorted(dataset_dir.glob("*.jsonl"), key=lambda item: item.name.lower()):
            if not path.is_file():
                continue
            dataset_name = path.name
            normalized_key = dataset_name.lower()
            # Prefer user dataset when name conflicts.
            if normalized_key in seen_names:
                continue
            seen_names.add(normalized_key)
            entries.append(
                {
                    "name": dataset_name,
                    "path": path,
                    "source": source,
                    "read_only": source != "user",
                }
            )
    entries.sort(key=lambda item: (0 if item["source"] == "user" else 1, item["name"].lower()))
    return entries


def _resolve_eval_dataset_path(dataset_name: str, current_user: Any) -> Optional[Dict[str, Any]]:
    normalized = _normalize_text(dataset_name)
    if not normalized:
        return None
    candidate_names = [normalized]
    if not normalized.lower().endswith(".jsonl"):
        candidate_names.append(f"{normalized}.jsonl")
    for source, dataset_dir in _get_eval_dataset_search_dirs(current_user):
        for name in candidate_names:
            path = dataset_dir / name
            if path.exists() and path.is_file() and path.suffix.lower() == ".jsonl":
                return {
                    "name": path.name,
                    "path": path,
                    "source": source,
                    "read_only": source != "user",
                }
    return None


def _get_eval_history_path() -> Path:
    return Path(__file__).resolve().parent.parent / "tests" / "eval_history.jsonl"


def _prune_eval_tasks(now_ts: Optional[float] = None) -> None:
    now = now_ts or time.time()
    stale_ids: List[str] = []
    for task_id, task in eval_tasks.items():
        status = task.get("status")
        finished_at = task.get("finished_at")
        if status not in {"completed", "failed", "canceled"}:
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


def _sha256_text(value: str) -> str:
    raw = (value or "").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _stable_json_hash(payload: Dict[str, Any]) -> str:
    dumped = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(dumped)


def _build_reproducibility_summary(
    payload: EvalRequest,
    *,
    total_rows: int,
    used_rows: int,
) -> Dict[str, Any]:
    dataset_hash = _sha256_text(payload.dataset_jsonl or "")
    config_payload = {
        "eval_scope": payload.eval_scope,
        "workspace": payload.workspace,
        "collection_id": payload.collection_id,
        "retrieval_mode": payload.retrieval_mode,
        "max_retrieval_docs": payload.max_retrieval_docs,
        "enable_ragas": payload.enable_ragas,
        "ragas_limit": payload.ragas_limit,
        "limit": payload.limit or 0,
        "cache_enabled": payload.cache_enabled,
        "cache_namespace": payload.cache_namespace,
        "ci_mode": payload.ci_mode,
        "fail_on_gate": payload.fail_on_gate,
        "baseline_history_id": payload.baseline_history_id,
        "classifier_model_path": payload.classifier_model_path,
        "classifier_positive_threshold": payload.classifier_positive_threshold,
        "classifier_negative_threshold": payload.classifier_negative_threshold,
    }
    config_hash = _stable_json_hash(config_payload)
    replay_key = f"{dataset_hash[:12]}-{config_hash[:12]}"
    return {
        "dataset_sha256": dataset_hash,
        "config_sha256": config_hash,
        "replay_key": replay_key,
        "dataset_total_rows": total_rows,
        "dataset_used_rows": used_rows,
        "collection_snapshot_id": payload.collection_id or "default_collection",
    }


def _disabled_lab(name: str, reason: str) -> Dict[str, Any]:
    return {
        "name": name,
        "enabled": False,
        "reason": reason,
    }


def _emit_progress(
    progress_callback,
    *,
    stage: str,
    percent: float,
    processed: int = 0,
    total: int = 0,
    message: str = "",
    lab: Optional[str] = None,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(
            {
                "stage": stage,
                "percent": max(0.0, min(100.0, float(percent))),
                "processed": int(max(0, processed)),
                "total": int(max(0, total)),
                "message": _normalize_text(message),
                "lab": _normalize_text(lab) or None,
            }
        )
    except Exception:
        # 进度回调失败不应影响主流程
        return


def _append_task_progress_event(task: Dict[str, Any], progress_payload: Dict[str, Any]) -> None:
    if not isinstance(task, dict):
        return
    events = task.setdefault("progress_events", [])
    normalized = {
        "ts": time.time(),
        "stage": _normalize_text(progress_payload.get("stage")) or "unknown",
        "percent": _safe_float(progress_payload.get("percent")),
        "processed": _safe_int(progress_payload.get("processed")) or 0,
        "total": _safe_int(progress_payload.get("total")) or 0,
        "message": _normalize_text(progress_payload.get("message")),
        "lab": _normalize_text(progress_payload.get("lab")) or None,
    }
    if events:
        last = events[-1]
        duplicate = (
            _normalize_text(last.get("stage")) == normalized["stage"]
            and (_safe_int(last.get("processed")) or 0) == normalized["processed"]
            and (_safe_int(last.get("total")) or 0) == normalized["total"]
            and _normalize_text(last.get("message")) == normalized["message"]
        )
        if duplicate:
            return
    events.append(normalized)
    max_events = max(20, EVAL_TASK_PROGRESS_EVENTS_MAX)
    if len(events) > max_events:
        del events[0:len(events) - max_events]


def _default_classifier_model_path() -> str:
    env_path = _normalize_text(os.getenv("RAG_RETRIEVAL_STAT_CLASSIFIER_MODEL_PATH"))
    if env_path:
        return env_path
    return str(
        Path(__file__).resolve().parent.parent
        / "agent"
        / "models"
        / "artifacts"
        / "retrieval_intent_nb_v1.json"
    )


def _evaluate_classifier_payload(
    payload: EvalRequest,
    current_user: Any,
    rows: List[Dict[str, Any]],
    total_rows: int,
    start_time: float,
    progress_callback=None,
) -> Dict[str, Any]:
    _emit_progress(
        progress_callback,
        stage="init",
        percent=5,
        processed=0,
        total=len(rows),
        message="初始化分类器实验室",
        lab="classifier",
    )
    model_path = _normalize_text(payload.classifier_model_path) or _default_classifier_model_path()
    classifier = RetrievalIntentClassifier(
        model_path=model_path,
        enabled=True,
        positive_threshold=payload.classifier_positive_threshold,
        negative_threshold=payload.classifier_negative_threshold,
    )
    if not classifier.ready:
        return {"error": f"分类器模型不可用: {classifier.error_message or model_path}"}

    _emit_progress(
        progress_callback,
        stage="running",
        percent=25,
        processed=0,
        total=len(rows),
        message="执行分类器推理与标注对比",
        lab="classifier",
    )
    classifier_lab_result = run_classifier_lab(rows, classifier)
    evaluated_items = classifier_lab_result.get("items") or []
    for item in evaluated_items:
        item["badcase"] = detect_badcase_tags(item)
    badcase_summary = build_badcase_summary(evaluated_items)
    classifier_summary = classifier_lab_result.get("summary") or {}
    classifier_quality_gate_summary = classifier_lab_result.get("quality_gate") or {}
    warning_messages: List[str] = []
    unlabeled_count = int(classifier_lab_result.get("unlabeled_count") or 0)
    unavailable_count = int(classifier_lab_result.get("classifier_unavailable_count") or 0)
    if unlabeled_count > 0:
        warning_messages.append(f"{unlabeled_count}条样本缺少need_retrieval标注，无法计入分类指标")
    if unavailable_count > 0:
        warning_messages.append(f"{unavailable_count}条样本分类器推理不可用")
    badcase_rate = _safe_float(badcase_summary.get("badcase_rate"))
    if badcase_rate is not None and badcase_rate >= 0.3:
        warning_messages.append("Badcase占比较高，建议优先排查FN和runtime_error样本")

    elapsed_ms = int((time.time() - start_time) * 1000)
    quality_summary = {
        "context_precision": None,
        "context_recall": None,
        "answer_relevancy": None,
        "faithfulness": None,
    }
    retrieval_summary = _build_retrieval_summary(evaluated_items, [])
    routing_summary = _build_routing_summary(evaluated_items)
    performance_summary = _build_performance_summary(evaluated_items, elapsed_ms)
    stability_summary = _build_stability_summary(evaluated_items)
    sop_summary = _build_sop_summary(evaluated_items)
    answer_overlap_summary = _build_answer_overlap_summary(evaluated_items)
    slice_summary = {
        "query_type": _build_slice_summary(evaluated_items, "query_type"),
        "difficulty": _build_slice_summary(evaluated_items, "difficulty"),
        "risk_level": _build_slice_summary(evaluated_items, "risk_level"),
    }
    quality_gate_summary = {
        "overall_status": "skipped",
        "pass_count": 0,
        "fail_count": 0,
        "skip_count": 0,
        "checks": [],
    }
    classifier_lab = build_classifier_lab_report(
        classifier_summary=classifier_summary,
        classifier_quality_gate=classifier_quality_gate_summary,
    )
    module_labs = {
        "classifier": classifier_lab,
        "retrieval": _disabled_lab("Retrieval Lab", "classifier_only_scope"),
        "generation": _disabled_lab("Generation Lab", "classifier_only_scope"),
        "medical_safety": _disabled_lab("Medical Safety Lab", "classifier_only_scope"),
    }
    release_decision = _build_release_decision(
        ci_mode=payload.ci_mode,
        fail_on_gate=payload.fail_on_gate,
        quality_gate_summary=classifier_quality_gate_summary,
        medical_safety_lab=None,
    )
    eval_labs = build_eval_labs_summary(
        quality_summary=quality_summary,
        retrieval_summary=retrieval_summary,
        routing_summary=routing_summary,
        stability_summary=stability_summary,
        sop_summary=sop_summary,
        answer_overlap_summary=answer_overlap_summary,
        classifier_summary=classifier_summary,
    )
    reproducibility_summary = _build_reproducibility_summary(
        payload,
        total_rows=total_rows,
        used_rows=len(rows),
    )
    performance_summary.update(
        {
            "retrieval_generation_elapsed_ms": 0,
            "ragas_elapsed_ms": 0,
            "non_retrieval_overhead_ms": elapsed_ms,
            "ragas_evaluated_rows": 0,
        }
    )
    cost_summary = _build_cost_summary(evaluated_items, elapsed_ms)
    cache_summary = {
        "enabled": payload.cache_enabled,
        "namespace": payload.cache_namespace,
        "hit_count": None,
        "miss_count": None,
        "hit_rate": None,
    }
    warning_text = "；".join(dict.fromkeys([msg for msg in warning_messages if msg])) or None
    _emit_progress(
        progress_callback,
        stage="summarizing",
        percent=90,
        processed=len(rows),
        total=len(rows),
        message="聚合分类器评测结果",
        lab="classifier",
    )
    run_report = {
        "run_id": str(uuid4()),
        "run_tag": payload.run_tag,
        "created_at": time.time(),
        "dataset": {
            "name": (payload.dataset_name or "").strip() or None,
            "total_rows": total_rows,
            "used_rows": len(rows),
            "request_size_bytes": len(payload.dataset_jsonl.encode("utf-8")) if payload.dataset_jsonl else 0,
        },
        "config": {
            "eval_scope": payload.eval_scope,
            "workspace": payload.workspace,
            "collection_id": payload.collection_id,
            "retrieval_mode": payload.retrieval_mode,
            "max_retrieval_docs": payload.max_retrieval_docs,
            "enable_ragas": False,
            "ragas_limit": 0,
            "limit": payload.limit or 0,
            "classifier_model_path": model_path,
            "classifier_positive_threshold": payload.classifier_positive_threshold,
            "classifier_negative_threshold": payload.classifier_negative_threshold,
            "classifier_model_version": classifier.model_version,
            "ci_mode": payload.ci_mode,
            "fail_on_gate": payload.fail_on_gate,
            "baseline_history_id": payload.baseline_history_id,
        },
        "quality_summary": quality_summary,
        "retrieval_summary": retrieval_summary,
        "routing_summary": routing_summary,
        "performance_summary": performance_summary,
        "stability_summary": stability_summary,
        "sop_summary": sop_summary,
        "answer_overlap_summary": answer_overlap_summary,
        "slice_summary": slice_summary,
        "quality_gate_summary": quality_gate_summary,
        "classifier_summary": classifier_summary,
        "classifier_quality_gate_summary": classifier_quality_gate_summary,
        "module_labs": module_labs,
        "eval_labs": eval_labs,
        "release_decision": release_decision,
        "badcase_summary": badcase_summary,
        "reproducibility_summary": reproducibility_summary,
        "cost_summary": cost_summary,
        "cache_summary": cache_summary,
    }
    include_items = bool(payload.include_item_details)
    response_payload = {
        "run": run_report,
        "summary": classifier_summary,
        "retrieval_fusion_summary": {},
        "retrieval_summary": retrieval_summary,
        "routing_summary": routing_summary,
        "performance_summary": performance_summary,
        "stability_summary": stability_summary,
        "sop_summary": sop_summary,
        "answer_overlap_summary": answer_overlap_summary,
        "slice_summary": slice_summary,
        "quality_gate_summary": quality_gate_summary,
        "classifier_summary": classifier_summary,
        "classifier_quality_gate_summary": classifier_quality_gate_summary,
        "module_labs": module_labs,
        "eval_labs": eval_labs,
        "release_decision": release_decision,
        "badcase_summary": badcase_summary,
        "reproducibility_summary": reproducibility_summary,
        "cost_summary": cost_summary,
        "cache_summary": cache_summary,
        "items": evaluated_items if include_items else [],
        "items_count": len(evaluated_items),
        "item_details_included": include_items,
        "elapsed_ms": elapsed_ms,
        "total": len(evaluated_items),
        "warning": warning_text,
    }
    response_payload["baseline_comparison"] = {"found": False, "reason": "classifier_only_scope"}
    response_payload["run"]["baseline_comparison"] = {"found": False, "reason": "classifier_only_scope"}
    try:
        history_result = dict(response_payload)
        history_result["items"] = _compact_items_for_history(evaluated_items)
        history_result["item_details_included"] = True
        history_result["items_count"] = len(evaluated_items)
        record = {
            "id": str(uuid4()),
            "created_at": time.time(),
            "user_id": str(current_user),
            "dataset_name": (payload.dataset_name or "").strip() or None,
            "dataset_total_lines": total_rows,
            "dataset_used_lines": len(rows),
            "dataset_size": len(payload.dataset_jsonl.encode("utf-8")) if payload.dataset_jsonl else 0,
            "workspace": payload.workspace,
            "eval_scope": payload.eval_scope,
            "retrieval_mode": payload.retrieval_mode,
            "max_retrieval_docs": payload.max_retrieval_docs,
            "collection_id": payload.collection_id,
            "enable_ragas": False,
            "run_tag": payload.run_tag,
            "cache_enabled": payload.cache_enabled,
            "cache_namespace": payload.cache_namespace,
            "limit": payload.limit or 0,
            "result": history_result,
        }
        _append_eval_history(record)
    except Exception as exc:
        logger.error(f"保存分类器评测历史失败: {exc}")
    if release_decision.get("blocked"):
        return {
            "error": "CI门禁失败：classifier实验室未通过",
            "result": response_payload,
        }
    _emit_progress(
        progress_callback,
        stage="completed",
        percent=100,
        processed=len(rows),
        total=len(rows),
        message="分类器评测完成",
        lab="classifier",
    )
    return response_payload


@router.get("/query")
async def query_rag(q: str):
    return {"query": q, "results": []}

@router.post("/index")
async def index_document(document: str):
    return {"status": "success", "document": document}


@router.post("/classifier-train")
async def train_intent_classifier(payload: TrainClassifierRequest, current_user: int = Depends(get_current_user)):
    if payload.split_strategy not in VALID_CLASSIFIER_SPLIT_STRATEGIES:
        return Response.error(f"不支持的split_strategy: {payload.split_strategy}")
    if payload.valid_ratio < 0 or payload.valid_ratio >= 1:
        return Response.error("valid_ratio必须在[0,1)范围内")
    if payload.test_ratio < 0 or payload.test_ratio >= 1:
        return Response.error("test_ratio必须在[0,1)范围内")
    if payload.valid_ratio + payload.test_ratio >= 1:
        return Response.error("valid_ratio + test_ratio 必须小于1")
    if payload.smoothing <= 0:
        return Response.error("smoothing必须大于0")
    if payload.min_token_freq <= 0:
        return Response.error("min_token_freq必须大于0")
    if payload.positive_threshold is not None and not (0 <= payload.positive_threshold <= 1):
        return Response.error("positive_threshold必须在[0,1]范围内")
    if payload.negative_threshold is not None and not (0 <= payload.negative_threshold <= 1):
        return Response.error("negative_threshold必须在[0,1]范围内")
    try:
        result = await run_in_threadpool(_train_classifier_payload, payload, current_user)
    except Exception as exc:
        return Response.error(f"训练失败: {str(exc)}")
    if result.get("error"):
        return Response.error(str(result.get("error")))
    return Response.success(result)


@router.post("/evaluate")
async def evaluate_rag(payload: EvalRequest, current_user: int = Depends(get_current_user)):
    response_payload = await _evaluate_payload(payload, current_user)
    if response_payload.get("error"):
        return Response.error(response_payload["error"])
    return Response.success(response_payload)


async def _evaluate_payload(payload: EvalRequest, current_user: int, progress_callback=None) -> Dict[str, Any]:
    start_time = time.time()
    _emit_progress(
        progress_callback,
        stage="init",
        percent=2,
        message="校验评测请求",
        lab=payload.eval_scope,
    )
    if payload.limit is not None and payload.limit < 0:
        return {"error": "limit不能为负数"}
    if payload.eval_scope not in VALID_EVAL_SCOPES:
        return {"error": f"不支持的eval_scope: {payload.eval_scope}"}
    if payload.classifier_positive_threshold < 0 or payload.classifier_positive_threshold > 1:
        return {"error": "classifier_positive_threshold必须在[0,1]范围内"}
    if payload.classifier_negative_threshold < 0 or payload.classifier_negative_threshold > 1:
        return {"error": "classifier_negative_threshold必须在[0,1]范围内"}
    if payload.eval_scope in {
        EvalScope.RAG_FULL,
        EvalScope.RETRIEVAL_ONLY,
        EvalScope.GENERATION_ONLY,
        EvalScope.MEDICAL_SAFETY_ONLY,
    }:
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
    _emit_progress(
        progress_callback,
        stage="dataset_ready",
        percent=8,
        processed=0,
        total=len(rows),
        message="数据集加载完成，准备执行",
        lab=payload.eval_scope,
    )
    if payload.eval_scope == EvalScope.CLASSIFIER_ONLY:
        return await run_in_threadpool(
            _evaluate_classifier_payload,
            payload,
            current_user,
            rows,
            total_rows,
            start_time,
            progress_callback,
        )
    scope = payload.eval_scope
    retrieval_only = scope == EvalScope.RETRIEVAL_ONLY
    generation_only = scope == EvalScope.GENERATION_ONLY
    medical_only = scope == EvalScope.MEDICAL_SAFETY_ONLY
    rag_full = scope == EvalScope.RAG_FULL
    _emit_progress(
        progress_callback,
        stage="running",
        percent=10,
        processed=0,
        total=len(rows),
        message="启动RAG评测执行",
        lab=scope,
    )
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
    for idx, row in enumerate(rows, start=1):
        question = _normalize_text(row.get("question") or row.get("query"))
        reference = _resolve_reference_text(row)
        expected_handoff = _to_optional_bool(
            row.get("expected_handoff")
            if "expected_handoff" in row
            else row.get("expect_handoff")
        )
        expected_triage_level = _normalize_text(
            row.get("expected_triage_level")
            if "expected_triage_level" in row
            else row.get("triage_level")
        ) or None
        gold_contexts = _normalize_text_list(
            row.get("gold_contexts")
            if "gold_contexts" in row
            else row.get("expected_contexts")
        )
        gold_sources = _normalize_text_list(
            row.get("gold_sources")
            if "gold_sources" in row
            else row.get("expected_sources")
        )
        query_type = _normalize_text(row.get("query_type") or row.get("intent_type")) or None
        difficulty = _normalize_text(row.get("difficulty")) or None
        risk_level = _normalize_text(row.get("risk_level") or row.get("medical_risk_level")) or None
        need_retrieval_label = parse_need_retrieval_label(row)
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
            "context_details": [],
            "contexts_count": 0,
            "contexts_total_chars": 0,
            "contexts_preview": [],
            "retrieval_sources": {},
            "retrieval_fusion_stats": {},
            "expected_handoff": expected_handoff,
            "expected_triage_level": expected_triage_level,
            "gold_contexts": gold_contexts,
            "gold_sources": gold_sources,
            "query_type": query_type,
            "difficulty": difficulty,
            "risk_level": risk_level,
            "need_retrieval_label": need_retrieval_label,
            "sop": {},
            "retrieval_label_eval": {},
            "classifier_label_eval": {},
            "routing": {},
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
                "context_details": eval_result.get("context_details") or [],
                "contexts_count": len(contexts),
                "contexts_total_chars": int(eval_result.get("contexts_total_chars") or 0),
                "contexts_preview": [str(c)[:300] for c in contexts[:3]],
                "retrieval_sources": eval_result.get("retrieval_sources") or {},
                "retrieval_fusion_stats": fusion_stats,
                "sop": eval_result.get("sop_trace") or {},
                "routing": eval_result.get("routing_trace") or {},
            })
            if isinstance(item_record.get("expected_handoff"), bool):
                handoff_required = bool((item_record.get("sop") or {}).get("handoff_required"))
                item_record["sop"]["handoff_expected"] = item_record["expected_handoff"]
                item_record["sop"]["handoff_correct"] = handoff_required == item_record["expected_handoff"]
            if item_record.get("expected_triage_level"):
                triage_level = _normalize_text((item_record.get("sop") or {}).get("triage_level")) or None
                expected_level = _normalize_text(item_record.get("expected_triage_level")) or None
                item_record["sop"]["triage_expected"] = expected_level
                item_record["sop"]["triage_correct"] = bool(
                    triage_level and expected_level and triage_level.lower() == expected_level.lower()
                )
            item_record["retrieval_label_eval"] = _evaluate_retrieval_labels(item_record)
            item_record["classifier_label_eval"] = build_item_classifier_label_eval(
                label=item_record.get("need_retrieval_label"),
                routing=item_record.get("routing") or {},
            )
        except Exception as exc:
            logger.error(f"单条评测失败: {exc}")
            item_record.update({
                "status": "error",
                "error": str(exc)
            })
            item_record["classifier_label_eval"] = build_item_classifier_label_eval(
                label=item_record.get("need_retrieval_label"),
                routing=item_record.get("routing") or {},
            )
        items.append(item_record)
        base = 12.0
        span = 58.0
        _emit_progress(
            progress_callback,
            stage="running",
            percent=base + span * (idx / max(len(rows), 1)),
            processed=idx,
            total=len(rows),
            message=f"已完成 {idx}/{len(rows)} 条样本",
            lab=scope,
        )
    if not items:
        return {"error": "评测数据中未找到有效问题"}
    retrieval_elapsed_ms = int((time.time() - start_time) * 1000)
    metrics_result: Dict[str, Any] = {"metrics": {}, "items": items}
    ragas_elapsed_ms = 0
    ragas_evaluated_rows = 0
    warning_messages: List[str] = []
    effective_enable_ragas = bool(payload.enable_ragas) and (rag_full or generation_only)
    if effective_enable_ragas:
        _emit_progress(
            progress_callback,
            stage="ragas",
            percent=75,
            processed=len(items),
            total=len(items),
            message="执行RAGAS语义评测",
            lab=scope,
        )
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
        if generation_only:
            warning_messages.append("Generation模块未启用RAGAS，仅返回规则层指标")
        elif retrieval_only or medical_only:
            warning_messages.append("当前模块默认关闭RAGAS，仅返回模块相关指标")
        else:
            warning_messages.append("已关闭RAGAS，仅返回基础指标")
    elapsed_ms = int((time.time() - start_time) * 1000)
    evaluated_items = metrics_result.get("items", items)
    fusion_summary = _aggregate_fusion_stats(fusion_stats_rows)
    retrieval_summary = _build_retrieval_summary(evaluated_items, fusion_stats_rows)
    routing_summary = _build_routing_summary(evaluated_items)
    performance_summary = _build_performance_summary(evaluated_items, elapsed_ms)
    stability_summary = _build_stability_summary(evaluated_items)
    sop_summary = _build_sop_summary(evaluated_items)
    answer_overlap_summary = _build_answer_overlap_summary(evaluated_items)
    slice_summary = {
        "query_type": _build_slice_summary(evaluated_items, "query_type"),
        "difficulty": _build_slice_summary(evaluated_items, "difficulty"),
        "risk_level": _build_slice_summary(evaluated_items, "risk_level")
    }
    classifier_summary = build_classifier_summary(evaluated_items)
    classifier_quality_gate_summary = build_classifier_quality_gate_summary(classifier_summary)
    classifier_lab = build_classifier_lab_report(
        classifier_summary=classifier_summary,
        classifier_quality_gate=classifier_quality_gate_summary,
    )
    retrieval_lab = build_retrieval_lab_report(
        evaluated_items,
        quality_summary=metrics_result.get("metrics", {}),
        hit_threshold=EVAL_CONTEXT_OVERLAP_HIT_THRESHOLD,
    )
    generation_lab = build_generation_lab_report(
        evaluated_items,
        quality_summary=metrics_result.get("metrics", {}),
    )
    medical_safety_lab = build_medical_safety_lab_report(evaluated_items)
    # 生成/安全模块更新了item内结构化诊断后，再计算badcase更稳定
    for item in evaluated_items:
        item["badcase"] = detect_badcase_tags(item)
    badcase_summary = build_badcase_summary(evaluated_items)
    quality_gate_summary = _build_quality_gate_summary(
        quality_summary=metrics_result.get("metrics", {}),
        retrieval_summary=retrieval_summary,
        performance_summary=performance_summary,
        stability_summary=stability_summary,
        sop_summary=sop_summary
    )
    safety_gate = (medical_safety_lab.get("hard_gate") or {}).get("overall_status")
    if rag_full:
        if safety_gate == "fail":
            quality_gate_summary = dict(quality_gate_summary)
            checks = list(quality_gate_summary.get("checks") or [])
            checks.append(
                {
                    "key": "medical_safety_hard_gate",
                    "label": "医疗安全硬门禁",
                    "operator": "==",
                    "value": "fail",
                    "threshold": "pass",
                    "status": "fail",
                }
            )
            quality_gate_summary["checks"] = checks
            quality_gate_summary["overall_status"] = "fail"
            quality_gate_summary["fail_count"] = int(quality_gate_summary.get("fail_count") or 0) + 1
        module_labs = {
            "classifier": classifier_lab,
            "retrieval": retrieval_lab,
            "generation": generation_lab,
            "medical_safety": medical_safety_lab,
        }
    elif retrieval_only:
        hit3 = _safe_float(((retrieval_lab.get("metrics") or {}).get("hit_at_k") or {}).get("3"))
        context_recall = _safe_float((retrieval_lab.get("metrics") or {}).get("context_recall"))
        checks = []
        checks.append(
            {
                "key": "retrieval_hit_at_3",
                "label": "Hit@3",
                "operator": ">=",
                "value": hit3,
                "threshold": _safe_float(os.getenv("RAG_EVAL_GATE_MIN_RETRIEVAL_HIT_AT_3", "0.7")),
                "status": "pass" if (hit3 is not None and hit3 >= _safe_float(os.getenv("RAG_EVAL_GATE_MIN_RETRIEVAL_HIT_AT_3", "0.7"))) else "fail",
            }
        )
        checks.append(
            {
                "key": "context_recall",
                "label": "上下文召回",
                "operator": ">=",
                "value": context_recall,
                "threshold": _safe_float(os.getenv("RAG_EVAL_GATE_MIN_CONTEXT_RECALL", "0.65")),
                "status": "pass" if (context_recall is not None and context_recall >= _safe_float(os.getenv("RAG_EVAL_GATE_MIN_CONTEXT_RECALL", "0.65"))) else "fail",
            }
        )
        fail_count = sum(1 for item in checks if item.get("status") == "fail")
        quality_gate_summary = {
            "overall_status": "pass" if fail_count == 0 else "fail",
            "pass_count": sum(1 for item in checks if item.get("status") == "pass"),
            "fail_count": fail_count,
            "skip_count": 0,
            "checks": checks,
        }
        module_labs = {
            "classifier": _disabled_lab("Classifier Lab", "retrieval_only_scope"),
            "retrieval": retrieval_lab,
            "generation": _disabled_lab("Generation Lab", "retrieval_only_scope"),
            "medical_safety": _disabled_lab("Medical Safety Lab", "retrieval_only_scope"),
        }
    elif generation_only:
        completeness = _safe_float((generation_lab.get("metrics") or {}).get("completeness"))
        citation_coverage = _safe_float((generation_lab.get("metrics") or {}).get("citation_coverage"))
        checks = []
        checks.append(
            {
                "key": "faithfulness",
                "label": "事实一致性",
                "operator": ">=",
                "value": _safe_float((generation_lab.get("metrics") or {}).get("faithfulness")),
                "threshold": _safe_float(os.getenv("RAG_EVAL_GATE_MIN_FAITHFULNESS", "0.65")),
                "status": "pass"
                if (
                    _safe_float((generation_lab.get("metrics") or {}).get("faithfulness")) is not None
                    and _safe_float((generation_lab.get("metrics") or {}).get("faithfulness"))
                    >= _safe_float(os.getenv("RAG_EVAL_GATE_MIN_FAITHFULNESS", "0.65"))
                )
                else "fail",
            }
        )
        checks.append(
            {
                "key": "completeness",
                "label": "答案完整性",
                "operator": ">=",
                "value": completeness,
                "threshold": _safe_float(os.getenv("RAG_EVAL_GATE_MIN_COMPLETENESS", "0.55")),
                "status": "pass" if (completeness is not None and completeness >= _safe_float(os.getenv("RAG_EVAL_GATE_MIN_COMPLETENESS", "0.55"))) else "fail",
            }
        )
        checks.append(
            {
                "key": "citation_coverage",
                "label": "引用覆盖率",
                "operator": ">=",
                "value": citation_coverage,
                "threshold": _safe_float(os.getenv("RAG_EVAL_GATE_MIN_CITATION_COVERAGE", "0.5")),
                "status": "pass"
                if (citation_coverage is not None and citation_coverage >= _safe_float(os.getenv("RAG_EVAL_GATE_MIN_CITATION_COVERAGE", "0.5")))
                else "fail",
            }
        )
        fail_count = sum(1 for item in checks if item.get("status") == "fail")
        quality_gate_summary = {
            "overall_status": "pass" if fail_count == 0 else "fail",
            "pass_count": sum(1 for item in checks if item.get("status") == "pass"),
            "fail_count": fail_count,
            "skip_count": 0,
            "checks": checks,
        }
        module_labs = {
            "classifier": _disabled_lab("Classifier Lab", "generation_only_scope"),
            "retrieval": _disabled_lab("Retrieval Lab", "generation_only_scope"),
            "generation": generation_lab,
            "medical_safety": _disabled_lab("Medical Safety Lab", "generation_only_scope"),
        }
    else:
        # medical_safety_only
        quality_gate_summary = medical_safety_lab.get("hard_gate") or {
            "overall_status": "skipped",
            "pass_count": 0,
            "fail_count": 0,
            "skip_count": 0,
            "checks": [],
        }
        module_labs = {
            "classifier": _disabled_lab("Classifier Lab", "medical_safety_only_scope"),
            "retrieval": _disabled_lab("Retrieval Lab", "medical_safety_only_scope"),
            "generation": _disabled_lab("Generation Lab", "medical_safety_only_scope"),
            "medical_safety": medical_safety_lab,
        }
    release_decision = _build_release_decision(
        ci_mode=payload.ci_mode,
        fail_on_gate=payload.fail_on_gate,
        quality_gate_summary=quality_gate_summary,
        medical_safety_lab=medical_safety_lab if rag_full or medical_only else None,
    )
    eval_labs = build_eval_labs_summary(
        quality_summary=metrics_result.get("metrics", {}),
        retrieval_summary=retrieval_summary,
        routing_summary=routing_summary,
        stability_summary=stability_summary,
        sop_summary=sop_summary,
        answer_overlap_summary=answer_overlap_summary,
        classifier_summary=classifier_summary,
    )
    reproducibility_summary = _build_reproducibility_summary(
        payload,
        total_rows=total_rows,
        used_rows=len(rows),
    )
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
    labeled_classifier_count = int(classifier_summary.get("labeled_items") or 0)
    if labeled_classifier_count == 0:
        warning_messages.append("当前数据集未提供need_retrieval标注，已跳过小模型分类指标")
    badcase_rate = _safe_float(badcase_summary.get("badcase_rate"))
    if badcase_rate is not None and badcase_rate >= 0.3:
        warning_messages.append("Badcase占比较高，建议查看样本标签并优先处理high级问题")
    if (medical_safety_lab.get("hard_gate") or {}).get("overall_status") == "fail":
        warning_messages.append("医疗安全硬门禁未通过，请先处理急危重症拦截与幻觉问题")
    if metrics_result.get("warning"):
        warning_messages.append(str(metrics_result.get("warning")))
    warning_text = "；".join(dict.fromkeys([msg for msg in warning_messages if msg])) or None
    _emit_progress(
        progress_callback,
        stage="summarizing",
        percent=92,
        processed=len(evaluated_items),
        total=len(evaluated_items),
        message="聚合实验室指标与门禁",
        lab=scope,
    )
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
            "eval_scope": payload.eval_scope,
            "workspace": payload.workspace,
            "collection_id": payload.collection_id,
            "retrieval_mode": payload.retrieval_mode,
            "max_retrieval_docs": payload.max_retrieval_docs,
            "enable_ragas": payload.enable_ragas,
            "effective_enable_ragas": effective_enable_ragas,
            "ragas_limit": payload.ragas_limit,
            "limit": payload.limit or 0,
            "classifier_model_path": _normalize_text(payload.classifier_model_path) or _default_classifier_model_path(),
            "classifier_positive_threshold": payload.classifier_positive_threshold,
            "classifier_negative_threshold": payload.classifier_negative_threshold,
            "ci_mode": payload.ci_mode,
            "fail_on_gate": payload.fail_on_gate,
            "baseline_history_id": payload.baseline_history_id,
        },
        "quality_summary": metrics_result.get("metrics", {}),
        "retrieval_summary": retrieval_summary,
        "routing_summary": routing_summary,
        "performance_summary": performance_summary,
        "stability_summary": stability_summary,
        "sop_summary": sop_summary,
        "answer_overlap_summary": answer_overlap_summary,
        "slice_summary": slice_summary,
        "quality_gate_summary": quality_gate_summary,
        "classifier_summary": classifier_summary,
        "classifier_quality_gate_summary": classifier_quality_gate_summary,
        "module_labs": module_labs,
        "eval_labs": eval_labs,
        "release_decision": release_decision,
        "badcase_summary": badcase_summary,
        "reproducibility_summary": reproducibility_summary,
        "cost_summary": cost_summary,
        "cache_summary": cache_summary
    }
    include_items = bool(payload.include_item_details)
    response_payload = {
        "run": run_report,
        "summary": metrics_result.get("metrics", {}),
        "retrieval_fusion_summary": fusion_summary,
        "retrieval_summary": retrieval_summary,
        "routing_summary": routing_summary,
        "performance_summary": performance_summary,
        "stability_summary": stability_summary,
        "sop_summary": sop_summary,
        "answer_overlap_summary": answer_overlap_summary,
        "slice_summary": slice_summary,
        "quality_gate_summary": quality_gate_summary,
        "classifier_summary": classifier_summary,
        "classifier_quality_gate_summary": classifier_quality_gate_summary,
        "module_labs": module_labs,
        "eval_labs": eval_labs,
        "release_decision": release_decision,
        "badcase_summary": badcase_summary,
        "reproducibility_summary": reproducibility_summary,
        "cost_summary": cost_summary,
        "cache_summary": cache_summary,
        "items": evaluated_items if include_items else [],
        "items_count": len(evaluated_items),
        "item_details_included": include_items,
        "elapsed_ms": elapsed_ms,
        "total": len(evaluated_items),
        "warning": warning_text
    }
    baseline_comparison = _build_baseline_comparison(current_user, payload, response_payload)
    response_payload["baseline_comparison"] = baseline_comparison
    response_payload["run"]["baseline_comparison"] = baseline_comparison
    try:
        history_result = dict(response_payload)
        history_result["items"] = _compact_items_for_history(evaluated_items)
        history_result["item_details_included"] = True
        history_result["items_count"] = len(evaluated_items)
        record = {
            "id": str(uuid4()),
            "created_at": time.time(),
            "user_id": str(current_user),
            "dataset_name": (payload.dataset_name or "").strip() or None,
            "dataset_total_lines": total_rows,
            "dataset_used_lines": len(rows),
            "dataset_size": len(payload.dataset_jsonl.encode("utf-8")) if payload.dataset_jsonl else 0,
            "workspace": payload.workspace,
            "eval_scope": payload.eval_scope,
            "retrieval_mode": payload.retrieval_mode,
            "max_retrieval_docs": payload.max_retrieval_docs,
            "collection_id": payload.collection_id,
            "enable_ragas": payload.enable_ragas,
            "run_tag": payload.run_tag,
            "cache_enabled": payload.cache_enabled,
            "cache_namespace": payload.cache_namespace,
            "limit": payload.limit or 0,
            "result": history_result
        }
        _append_eval_history(record)
    except Exception as exc:
        logger.error(f"保存评测历史失败: {exc}")
    if release_decision.get("blocked"):
        return {
            "error": "CI门禁失败：质量门禁或医疗安全门禁未通过",
            "result": response_payload,
        }
    _emit_progress(
        progress_callback,
        stage="completed",
        percent=100,
        processed=len(evaluated_items),
        total=len(evaluated_items),
        message="评测完成",
        lab=scope,
    )
    return response_payload


async def _run_eval_task(task_id: str, payload: EvalRequest, current_user: int) -> None:
    task = eval_tasks.get(task_id)
    if not isinstance(task, dict):
        return
    task["status"] = "running"
    task["started_at"] = time.time()
    task["progress"] = {
        "stage": "running",
        "percent": 1.0,
        "processed": 0,
        "total": 0,
        "message": "任务已启动",
        "lab": payload.eval_scope,
    }
    task["progress_events"] = list(task.get("progress_events") or [])
    _append_task_progress_event(task, task["progress"])

    def _progress_callback(progress: Dict[str, Any]) -> None:
        if task_id not in eval_tasks:
            return
        eval_tasks[task_id]["progress"] = progress
        _append_task_progress_event(eval_tasks[task_id], progress)

    try:
        result = await _evaluate_payload(payload, current_user, _progress_callback)
        if result.get("error"):
            failed_payload = {
                "status": "failed",
                "error": result.get("error"),
                "result": result.get("result"),
                "finished_at": time.time(),
                "progress": {
                    "stage": "failed",
                    "percent": 100.0,
                    "processed": 0,
                    "total": 0,
                    "message": _normalize_text(result.get("error")),
                    "lab": payload.eval_scope,
                },
            }
            eval_tasks[task_id].update(failed_payload)
            _append_task_progress_event(eval_tasks[task_id], failed_payload["progress"])
            return
        done_payload = {
            "status": "completed",
            "result": result,
            "finished_at": time.time(),
            "progress": {
                "stage": "completed",
                "percent": 100.0,
                "processed": int(result.get("items_count") or result.get("total") or 0),
                "total": int(result.get("total") or result.get("items_count") or 0),
                "message": "评测完成",
                "lab": payload.eval_scope,
            },
        }
        eval_tasks[task_id].update(done_payload)
        _append_task_progress_event(eval_tasks[task_id], done_payload["progress"])
    except asyncio.CancelledError:
        canceled_task = eval_tasks.get(task_id)
        if isinstance(canceled_task, dict):
            if canceled_task.get("status") != "canceled":
                cancel_payload = {
                    "status": "canceled",
                    "error": None,
                    "finished_at": time.time(),
                    "progress": {
                        "stage": "canceled",
                        "percent": 100.0,
                        "processed": _safe_int((canceled_task.get("progress") or {}).get("processed")) or 0,
                        "total": _safe_int((canceled_task.get("progress") or {}).get("total")) or 0,
                        "message": "任务已取消",
                        "lab": payload.eval_scope,
                    },
                }
                canceled_task.update(cancel_payload)
                _append_task_progress_event(canceled_task, cancel_payload["progress"])
        return
    except Exception as exc:
        exception_payload = {
            "status": "failed",
            "error": str(exc),
            "finished_at": time.time(),
            "progress": {
                "stage": "failed",
                "percent": 100.0,
                "processed": 0,
                "total": 0,
                "message": str(exc),
                "lab": payload.eval_scope,
            },
        }
        eval_tasks[task_id].update(exception_payload)
        _append_task_progress_event(eval_tasks[task_id], exception_payload["progress"])


@router.post("/evaluate-async")
async def evaluate_rag_async(payload: EvalRequest, current_user: int = Depends(get_current_user)):
    _prune_eval_tasks()
    task_id = str(uuid4())
    submitted_at = time.time()
    queued_progress = {
        "stage": "queued",
        "percent": 0.0,
        "processed": 0,
        "total": 0,
        "message": "任务排队中",
        "lab": payload.eval_scope,
    }
    eval_tasks[task_id] = {
        "status": "queued",
        "user_id": str(current_user),
        "eval_scope": payload.eval_scope,
        "submitted_at": submitted_at,
        "result": None,
        "error": None,
        "progress": queued_progress,
        "progress_events": [
            {
                "ts": submitted_at,
                "stage": "queued",
                "percent": 0.0,
                "processed": 0,
                "total": 0,
                "message": "任务排队中",
                "lab": payload.eval_scope,
            }
        ],
    }
    runner = asyncio.create_task(_run_eval_task(task_id, payload, current_user))
    eval_tasks[task_id]["runner"] = runner
    return Response.success({"task_id": task_id})


@router.post("/evaluate-cancel/{task_id}")
async def cancel_rag_evaluation(task_id: str, current_user: int = Depends(get_current_user)):
    _prune_eval_tasks()
    task = eval_tasks.get(task_id)
    if not task or str(task.get("user_id")) != str(current_user):
        return Response.error("评测任务不存在")
    status = _normalize_text(task.get("status")).lower()
    if status in {"completed", "failed", "canceled"}:
        return Response.success({"task_id": task_id, "status": status, "message": "任务已结束，无需取消"})

    progress = task.get("progress") or {}
    cancel_progress = {
        "stage": "canceled",
        "percent": 100.0,
        "processed": _safe_int(progress.get("processed")) or 0,
        "total": _safe_int(progress.get("total")) or 0,
        "message": "任务已取消",
        "lab": _normalize_text(progress.get("lab")) or _normalize_text(task.get("eval_scope")) or None,
    }
    task.update(
        {
            "status": "canceled",
            "error": None,
            "finished_at": time.time(),
            "progress": cancel_progress,
        }
    )
    _append_task_progress_event(task, cancel_progress)

    runner = task.get("runner")
    if isinstance(runner, asyncio.Task) and not runner.done():
        runner.cancel()

    return Response.success({"task_id": task_id, "status": "canceled"})


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
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "error": task.get("error"),
        "progress": task.get("progress"),
        "progress_events": task.get("progress_events") or [],
    }
    now_ts = time.time()
    started_at = _safe_float(task.get("started_at"))
    submitted_at = _safe_float(task.get("submitted_at"))
    finished_at = _safe_float(task.get("finished_at"))
    if started_at is not None:
        runtime_end = finished_at if finished_at is not None else now_ts
        payload["runtime_seconds"] = round(max(0.0, runtime_end - started_at), 3)
    if started_at is not None and submitted_at is not None:
        payload["queue_wait_seconds"] = round(max(0.0, started_at - submitted_at), 3)
    if task.get("status") == "completed":
        payload["result"] = task.get("result")
    elif task.get("status") in {"failed", "canceled"} and isinstance(task.get("result"), dict):
        payload["result"] = task.get("result")
    return Response.success(payload)


@router.get("/eval-datasets")
async def list_eval_datasets(current_user: int = Depends(get_current_user)):
    items: List[Dict[str, Any]] = []
    for entry in _list_eval_dataset_entries(current_user):
        path = entry.get("path")
        if not isinstance(path, Path):
            continue
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
            "size": size,
            "source": entry.get("source") or "user",
            "read_only": bool(entry.get("read_only")),
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
    resolved = _resolve_eval_dataset_path(dataset_name, current_user)
    if not resolved:
        return Response.error("评测数据集不存在")
    path = resolved["path"]
    content = path.read_text(encoding="utf-8")
    return Response.success({
        "name": resolved.get("name") or dataset_name,
        "content": content,
        "source": resolved.get("source") or "user",
        "read_only": bool(resolved.get("read_only")),
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
    handoff_count = 0
    emergency_count = 0
    expected_handoff_total = 0
    handoff_correct_count = 0
    red_flag_total = 0
    routing_stage_counter: Dict[str, int] = {}
    routing_llm_fallback_count = 0
    routing_statistical_stage_count = 0
    classifier_labeled_count = 0
    classifier_correct_count = 0
    classifier_tp = 0
    classifier_fp = 0
    classifier_tn = 0
    classifier_fn = 0
    classifier_uncertain_count = 0
    classifier_fallback_probability_count = 0
    badcase_count = 0
    badcase_high_count = 0
    badcase_medium_count = 0
    badcase_tag_counter: Dict[str, int] = {}
    for item in all_items:
        sop = item.get("sop") or {}
        routing = item.get("routing") or {}
        classifier_eval = item.get("classifier_label_eval") or {}
        badcase = item.get("badcase")
        if not isinstance(badcase, dict):
            badcase = detect_badcase_tags(item)
            item["badcase"] = badcase
        routing_stage = _normalize_text(routing.get("stage")) or "unknown"
        routing_stage_counter[routing_stage] = routing_stage_counter.get(routing_stage, 0) + 1
        if routing_stage == "llm":
            routing_llm_fallback_count += 1
        if routing_stage == "statistical_classifier":
            routing_statistical_stage_count += 1
        if bool(sop.get("handoff_required")):
            handoff_count += 1
        if _normalize_text(sop.get("triage_level")).lower() == "emergency":
            emergency_count += 1
        red_flags = sop.get("red_flags") or []
        if isinstance(red_flags, list):
            red_flag_total += len(red_flags)
        if isinstance(item.get("expected_handoff"), bool):
            expected_handoff_total += 1
            if bool(sop.get("handoff_required")) == bool(item.get("expected_handoff")):
                handoff_correct_count += 1
        if bool(classifier_eval.get("has_label")):
            classifier_labeled_count += 1
            if classifier_eval.get("is_correct") is True:
                classifier_correct_count += 1
        confusion = _normalize_text(classifier_eval.get("confusion")).lower()
        if confusion == "tp":
            classifier_tp += 1
        elif confusion == "fp":
            classifier_fp += 1
        elif confusion == "tn":
            classifier_tn += 1
        elif confusion == "fn":
            classifier_fn += 1
        if classifier_eval.get("predicted") is None:
            classifier_uncertain_count += 1
        if bool(classifier_eval.get("predicted_from_probability_fallback")):
            classifier_fallback_probability_count += 1
        if bool(badcase.get("is_badcase")):
            badcase_count += 1
        badcase_severity = _normalize_text(badcase.get("severity")).lower()
        if badcase_severity == "high":
            badcase_high_count += 1
        elif badcase_severity == "medium":
            badcase_medium_count += 1
        badcase_tags = badcase.get("tags") or []
        if isinstance(badcase_tags, list):
            for tag in badcase_tags:
                text = _normalize_text(tag)
                if not text:
                    continue
                badcase_tag_counter[text] = badcase_tag_counter.get(text, 0) + 1

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
            "avg_contexts_count": round(sum(contexts_counts) / len(contexts_counts), 2) if contexts_counts else None,
            "handoff_count": handoff_count,
            "emergency_count": emergency_count,
            "handoff_rate": _safe_float(_safe_divide(handoff_count, total_items)),
            "avg_red_flags_per_item": round(red_flag_total / total_items, 4) if total_items else None,
            "expected_handoff_count": expected_handoff_total,
            "handoff_accuracy": _safe_float(_safe_divide(handoff_correct_count, expected_handoff_total))
            if expected_handoff_total > 0
            else None,
            "routing_stage_distribution": routing_stage_counter,
            "routing_llm_fallback_rate": _safe_float(_safe_divide(routing_llm_fallback_count, total_items)),
            "routing_statistical_stage_rate": _safe_float(_safe_divide(routing_statistical_stage_count, total_items)),
            "classifier_labeled_count": classifier_labeled_count,
            "classifier_accuracy": _safe_float(_safe_divide(classifier_correct_count, classifier_labeled_count))
            if classifier_labeled_count > 0
            else None,
            "classifier_uncertain_rate": _safe_float(_safe_divide(classifier_uncertain_count, total_items)),
            "classifier_fallback_probability_rate": _safe_float(
                _safe_divide(classifier_fallback_probability_count, total_items)
            ),
            "classifier_tp": classifier_tp,
            "classifier_fp": classifier_fp,
            "classifier_tn": classifier_tn,
            "classifier_fn": classifier_fn,
            "badcase_count": badcase_count,
            "badcase_rate": _safe_float(_safe_divide(badcase_count, total_items)),
            "badcase_high_count": badcase_high_count,
            "badcase_medium_count": badcase_medium_count,
            "badcase_tag_distribution": badcase_tag_counter,
        }
    })


@router.get("/eval-history/{history_id}/badcases")
async def get_eval_history_badcases(
    history_id: str,
    page: int = 1,
    page_size: int = 10,
    tag: Optional[str] = None,
    severity: Optional[str] = None,
    current_user: int = Depends(get_current_user),
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

    target_tag = _normalize_text(tag).lower() or None
    target_severity = _normalize_text(severity).lower() or None
    badcases: List[Dict[str, Any]] = []
    for item in all_items:
        badcase = item.get("badcase")
        if not isinstance(badcase, dict):
            badcase = detect_badcase_tags(item)
            item["badcase"] = badcase
        if not bool(badcase.get("is_badcase")):
            continue
        severity_text = _normalize_text(badcase.get("severity")).lower()
        tags = [str(t).strip().lower() for t in (badcase.get("tags") or []) if str(t).strip()]
        if target_severity and severity_text != target_severity:
            continue
        if target_tag and target_tag not in tags:
            continue
        badcases.append(item)

    total_items = len(badcases)
    total_pages = max(1, (total_items + page_size - 1) // page_size) if total_items > 0 else 1
    page = min(page, total_pages)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_items = badcases[start_idx:end_idx]

    return Response.success(
        {
            "history_id": history_id,
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
            "items": page_items,
            "filters": {
                "tag": target_tag,
                "severity": target_severity,
            },
            "summary": build_badcase_summary(badcases),
        }
    )
