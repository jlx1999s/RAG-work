from typing import Any, Dict, List, Optional


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


def build_classifier_lab_report(
    classifier_summary: Dict[str, Any],
    classifier_quality_gate: Dict[str, Any],
) -> Dict[str, Any]:
    labeled = int(classifier_summary.get("labeled_items") or 0)
    accuracy = _safe_float(classifier_summary.get("accuracy"))
    f1 = _safe_float(classifier_summary.get("f1"))
    recall = _safe_float(classifier_summary.get("recall"))
    uncertain_rate = _safe_float(classifier_summary.get("uncertain_rate"))
    fallback_prob_rate = _safe_float(classifier_summary.get("fallback_probability_rate"))

    recommendation = {
        "replace_lightweight_classifier": False,
        "confidence": "low",
        "reason": "标注样本不足或关键指标未达标",
        "fallback_llm_when": [
            "statistical_classifier_uncertain",
            "statistical_classifier_unavailable",
        ],
    }

    strong = (
        labeled >= 30
        and f1 is not None
        and recall is not None
        and uncertain_rate is not None
        and f1 >= 0.82
        and recall >= 0.85
        and uncertain_rate <= 0.25
    )
    medium = (
        labeled >= 15
        and f1 is not None
        and recall is not None
        and f1 >= 0.75
        and recall >= 0.78
    )

    if strong:
        recommendation = {
            "replace_lightweight_classifier": True,
            "confidence": "high",
            "reason": "F1/Recall/不确定率均达到可替代阈值",
            "fallback_llm_when": [
                "statistical_classifier_uncertain",
                "statistical_classifier_unavailable",
                "medical_safety_short_circuit",
            ],
        }
    elif medium:
        recommendation = {
            "replace_lightweight_classifier": False,
            "confidence": "medium",
            "reason": "性能可用但仍建议保留轻量分类器或逐步灰度替换",
            "fallback_llm_when": [
                "statistical_classifier_uncertain",
                "statistical_classifier_unavailable",
            ],
        }

    return {
        "name": "Classifier Lab",
        "enabled": True,
        "summary": classifier_summary,
        "quality_gate": classifier_quality_gate,
        "evidence": {
            "labeled_items": labeled,
            "accuracy": accuracy,
            "f1": f1,
            "recall_at_positive": recall,
            "uncertain_rate": uncertain_rate,
            "fallback_probability_rate": fallback_prob_rate,
            "error_profile": {
                "fp": int(classifier_summary.get("fp") or 0),
                "fn": int(classifier_summary.get("fn") or 0),
                "tp": int(classifier_summary.get("tp") or 0),
                "tn": int(classifier_summary.get("tn") or 0),
            },
            "positive_rate": _safe_divide(
                int(classifier_summary.get("tp") or 0) + int(classifier_summary.get("fn") or 0),
                labeled,
            ),
        },
        "decision": recommendation,
    }

