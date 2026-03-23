import os
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
    if text in {"true", "1", "yes", "y", "是", "need", "需要"}:
        return True
    if text in {"false", "0", "no", "n", "否", "noneed", "不需要"}:
        return False
    return None


def parse_need_retrieval_label(row: Dict[str, Any]) -> Optional[bool]:
    candidate_keys = [
        "need_retrieval",
        "expected_need_retrieval",
        "need_retrieval_gold",
        "label",
    ]
    for key in candidate_keys:
        if key not in row:
            continue
        parsed = _to_optional_bool(row.get(key))
        if parsed is not None:
            return parsed
    return None


def build_item_classifier_label_eval(
    label: Optional[bool],
    routing: Dict[str, Any],
    *,
    predicted_from_probability_fallback: bool = False,
) -> Dict[str, Any]:
    predicted = routing.get("need_retrieval")
    predicted_bool = predicted if isinstance(predicted, bool) else None
    stage = _normalize_text(routing.get("stage")) or "unknown"
    confidence = _safe_float((routing.get("statistical_classifier") or {}).get("confidence"))
    probability = _safe_float((routing.get("statistical_classifier") or {}).get("probability"))

    confusion = None
    is_correct = None
    if isinstance(label, bool) and isinstance(predicted_bool, bool):
        if predicted_bool and label:
            confusion = "tp"
        elif predicted_bool and not label:
            confusion = "fp"
        elif (not predicted_bool) and (not label):
            confusion = "tn"
        else:
            confusion = "fn"
        is_correct = predicted_bool == label

    return {
        "has_label": isinstance(label, bool),
        "label": label if isinstance(label, bool) else None,
        "predicted": predicted_bool,
        "prediction_stage": stage,
        "predicted_from_probability_fallback": bool(predicted_from_probability_fallback),
        "probability": probability,
        "confidence": confidence,
        "is_correct": is_correct,
        "confusion": confusion,
    }


def build_classifier_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_items = len(items)
    stage_distribution: Dict[str, int] = {}
    labeled_count = 0
    evaluated_count = 0
    correct_count = 0
    tp = fp = tn = fn = 0
    uncertain_count = 0
    fallback_count = 0

    for item in items:
        eval_row = item.get("classifier_label_eval") or {}
        stage = _normalize_text(eval_row.get("prediction_stage")) or "unknown"
        stage_distribution[stage] = stage_distribution.get(stage, 0) + 1

        if bool(eval_row.get("predicted_from_probability_fallback")):
            fallback_count += 1
        if eval_row.get("predicted") is None:
            uncertain_count += 1

        if not bool(eval_row.get("has_label")):
            continue
        labeled_count += 1
        confusion = _normalize_text(eval_row.get("confusion")).lower()
        if confusion in {"tp", "fp", "tn", "fn"}:
            evaluated_count += 1
            if confusion == "tp":
                tp += 1
            elif confusion == "fp":
                fp += 1
            elif confusion == "tn":
                tn += 1
            else:
                fn += 1
        if eval_row.get("is_correct") is True:
            correct_count += 1

    precision = _safe_divide(tp, tp + fp) if (tp + fp) > 0 else None
    recall = _safe_divide(tp, tp + fn) if (tp + fn) > 0 else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    accuracy = _safe_divide(correct_count, labeled_count) if labeled_count > 0 else None
    uncertain_rate = _safe_divide(uncertain_count, total_items) if total_items > 0 else None
    fallback_rate = _safe_divide(fallback_count, total_items) if total_items > 0 else None

    return {
        "total_items": total_items,
        "labeled_items": labeled_count,
        "evaluated_items": evaluated_count,
        "correct_items": correct_count,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy, 6) if accuracy is not None else None,
        "precision": round(precision, 6) if precision is not None else None,
        "recall": round(recall, 6) if recall is not None else None,
        "f1": round(f1, 6) if f1 is not None else None,
        "uncertain_count": uncertain_count,
        "uncertain_rate": round(uncertain_rate, 6) if uncertain_rate is not None else None,
        "fallback_probability_count": fallback_count,
        "fallback_probability_rate": round(fallback_rate, 6) if fallback_rate is not None else None,
        "stage_distribution": stage_distribution,
    }


def build_classifier_quality_gate_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    min_f1 = _safe_float(os.getenv("RAG_EVAL_GATE_MIN_CLASSIFIER_F1", "0.78"))
    min_recall = _safe_float(os.getenv("RAG_EVAL_GATE_MIN_CLASSIFIER_RECALL", "0.8"))
    max_uncertain = _safe_float(os.getenv("RAG_EVAL_GATE_MAX_CLASSIFIER_UNCERTAIN_RATE", "0.4"))

    rules = [
        {
            "key": "classifier_f1",
            "label": "分类器F1",
            "operator": ">=",
            "value": _safe_float(summary.get("f1")),
            "threshold": min_f1,
        },
        {
            "key": "classifier_recall",
            "label": "分类器召回率",
            "operator": ">=",
            "value": _safe_float(summary.get("recall")),
            "threshold": min_recall,
        },
        {
            "key": "classifier_uncertain_rate",
            "label": "分类器不确定率",
            "operator": "<=",
            "value": _safe_float(summary.get("uncertain_rate")),
            "threshold": max_uncertain,
        },
    ]
    checks: List[Dict[str, Any]] = []
    pass_count = fail_count = skip_count = 0

    for rule in rules:
        value = rule["value"]
        threshold = rule["threshold"]
        status = "skipped"
        if value is not None and threshold is not None:
            if rule["operator"] == "<=":
                status = "pass" if value <= threshold else "fail"
            else:
                status = "pass" if value >= threshold else "fail"
        if status == "pass":
            pass_count += 1
        elif status == "fail":
            fail_count += 1
        else:
            skip_count += 1
        checks.append(
            {
                "key": rule["key"],
                "label": rule["label"],
                "operator": rule["operator"],
                "value": value,
                "threshold": threshold,
                "status": status,
            }
        )
    overall = "pass" if fail_count == 0 else "fail"
    return {
        "overall_status": overall,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "skip_count": skip_count,
        "checks": checks,
    }


def build_eval_labs_summary(
    *,
    quality_summary: Dict[str, Any],
    retrieval_summary: Dict[str, Any],
    routing_summary: Dict[str, Any],
    stability_summary: Dict[str, Any],
    sop_summary: Dict[str, Any],
    answer_overlap_summary: Dict[str, Any],
    classifier_summary: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "classifier_validation": {
            "enabled": bool(classifier_summary),
            "summary": classifier_summary or {},
        },
        "rag_retrieval_validation": {
            "context_precision": _safe_float(quality_summary.get("context_precision")),
            "context_recall": _safe_float(quality_summary.get("context_recall")),
            "context_presence_rate": _safe_float(retrieval_summary.get("context_presence_rate")),
            "labeled_context_hit_rate": _safe_float(retrieval_summary.get("labeled_context_hit_rate")),
            "labeled_source_hit_rate": _safe_float(retrieval_summary.get("labeled_source_hit_rate")),
        },
        "rag_generation_validation": {
            "answer_relevancy": _safe_float(quality_summary.get("answer_relevancy")),
            "faithfulness": _safe_float(quality_summary.get("faithfulness")),
            "answer_overlap_f1": _safe_float(answer_overlap_summary.get("avg_answer_overlap_f1")),
            "abstained_answer_rate": _safe_float(stability_summary.get("abstained_answer_rate")),
        },
        "medical_safety_validation": {
            "handoff_rate": _safe_float(sop_summary.get("handoff_rate")),
            "handoff_accuracy": _safe_float(sop_summary.get("handoff_accuracy")),
            "handoff_recall": _safe_float(sop_summary.get("handoff_recall")),
            "structured_decision_valid_rate": _safe_float(sop_summary.get("structured_decision_valid_rate")),
            "llm_fallback_rate": _safe_float(routing_summary.get("llm_fallback_rate")),
            "error_rate": _safe_float(stability_summary.get("error_rate")),
        },
    }


def run_classifier_lab(
    rows: List[Dict[str, Any]],
    classifier: Any,
) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    unlabeled_count = 0
    unavailable_count = 0

    for row in rows:
        question = _normalize_text(row.get("question") or row.get("query"))
        if not question:
            continue
        label = parse_need_retrieval_label(row)
        if label is None:
            unlabeled_count += 1
        classifier_result = classifier.predict(question)
        available = bool(classifier_result.get("available"))
        if not available:
            unavailable_count += 1
        decision = classifier_result.get("decision")
        probability = _safe_float(classifier_result.get("probability"))
        used_fallback = False
        if decision is None and probability is not None:
            decision = probability >= 0.5
            used_fallback = True

        routing_trace = {
            "need_retrieval": decision,
            "stage": "statistical_classifier",
            "reason": _normalize_text(classifier_result.get("reason")),
            "statistical_classifier": {
                "available": classifier_result.get("available"),
                "decision": classifier_result.get("decision"),
                "probability": probability,
                "confidence": _safe_float(classifier_result.get("confidence")),
                "band": _normalize_text(classifier_result.get("band")),
                "model_version": _normalize_text(classifier_result.get("model_version")),
                "thresholds": classifier_result.get("thresholds") or {},
            },
        }
        item = {
            "question": question,
            "need_retrieval_label": label,
            "routing": routing_trace,
            "status": "success" if available else "error",
            "error": None if available else _normalize_text(classifier_result.get("reason")),
            "latency_ms": None,
            "contexts_count": 0,
            "contexts_total_chars": 0,
            "contexts_preview": [],
            "sop": {},
            "metrics": {
                "context_precision": None,
                "context_recall": None,
                "answer_relevancy": None,
                "faithfulness": None,
            },
        }
        item["classifier_label_eval"] = build_item_classifier_label_eval(
            label=label,
            routing=routing_trace,
            predicted_from_probability_fallback=used_fallback,
        )
        items.append(item)

    summary = build_classifier_summary(items)
    quality_gate = build_classifier_quality_gate_summary(summary)
    return {
        "items": items,
        "summary": summary,
        "quality_gate": quality_gate,
        "unlabeled_count": unlabeled_count,
        "classifier_unavailable_count": unavailable_count,
    }

