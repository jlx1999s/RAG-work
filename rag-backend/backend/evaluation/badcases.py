from collections import Counter
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


def _is_abstained_answer(answer: Any) -> bool:
    text = _normalize_text(answer).lower()
    if not text:
        return True
    markers = [
        "未在给定上下文中找到答案",
        "无法根据提供的上下文",
        "根据提供的上下文无法",
        "not found in the provided context",
        "cannot answer from the provided context",
    ]
    return any(marker in text for marker in markers)


def _contains_any(tags: List[str], candidates: set[str]) -> bool:
    return any(tag in candidates for tag in tags)


def detect_badcase_tags(
    item: Dict[str, Any],
    *,
    min_faithfulness: float = 0.55,
    min_answer_relevancy: float = 0.55,
) -> Dict[str, Any]:
    tags: List[str] = []
    reasons: List[str] = []
    status = _normalize_text(item.get("status")).lower() or "success"
    metrics = item.get("metrics") or {}
    routing = item.get("routing") or {}
    sop = item.get("sop") or {}
    retrieval_label_eval = item.get("retrieval_label_eval") or {}
    classifier_label_eval = item.get("classifier_label_eval") or {}

    if status == "error":
        tags.append("runtime_error")
        reasons.append("单条样本执行失败")

    if _is_abstained_answer(item.get("answer")):
        tags.append("abstained_answer")
        reasons.append("回答为拒答或空回答")

    contexts_count = int(item.get("contexts_count") or 0)
    if bool(routing.get("need_retrieval")) and contexts_count <= 0:
        tags.append("retrieval_empty")
        reasons.append("判定需要检索但未返回上下文")

    if bool(retrieval_label_eval.get("has_gold_contexts")) and not bool(retrieval_label_eval.get("gold_context_hit")):
        tags.append("gold_context_miss")
        reasons.append("未命中标注gold_contexts")

    if bool(retrieval_label_eval.get("has_gold_sources")) and not bool(retrieval_label_eval.get("gold_source_hit")):
        tags.append("gold_source_miss")
        reasons.append("未命中标注gold_sources")

    expected_handoff = item.get("expected_handoff")
    if isinstance(expected_handoff, bool):
        handoff_required = bool(sop.get("handoff_required"))
        if expected_handoff and not handoff_required:
            tags.append("safety_false_negative")
            reasons.append("标注应转人工，但实际未转人工")
        elif (not expected_handoff) and handoff_required:
            tags.append("safety_false_positive")
            reasons.append("标注不应转人工，但实际触发转人工")

    confusion = _normalize_text(classifier_label_eval.get("confusion")).lower()
    if confusion == "fn":
        tags.append("classifier_false_negative")
        reasons.append("检索意图分类漏判（FN）")
    elif confusion == "fp":
        tags.append("classifier_false_positive")
        reasons.append("检索意图分类误判（FP）")

    faithfulness = _safe_float(metrics.get("faithfulness"))
    if faithfulness is not None and faithfulness < float(min_faithfulness):
        tags.append("low_faithfulness")
        reasons.append(f"事实一致性低于阈值({min_faithfulness})")

    answer_relevancy = _safe_float(metrics.get("answer_relevancy"))
    if answer_relevancy is not None and answer_relevancy < float(min_answer_relevancy):
        tags.append("low_answer_relevancy")
        reasons.append(f"答案相关性低于阈值({min_answer_relevancy})")

    unique_tags: List[str] = []
    for tag in tags:
        if tag not in unique_tags:
            unique_tags.append(tag)

    high_tags = {
        "runtime_error",
        "safety_false_negative",
        "classifier_false_negative",
        "retrieval_empty",
    }
    medium_tags = {
        "gold_context_miss",
        "gold_source_miss",
        "classifier_false_positive",
        "safety_false_positive",
        "low_faithfulness",
        "abstained_answer",
    }

    severity = "none"
    if _contains_any(unique_tags, high_tags):
        severity = "high"
    elif _contains_any(unique_tags, medium_tags):
        severity = "medium"
    elif unique_tags:
        severity = "low"

    priority_order = [
        "runtime_error",
        "safety_false_negative",
        "classifier_false_negative",
        "retrieval_empty",
        "gold_context_miss",
        "low_faithfulness",
        "abstained_answer",
        "classifier_false_positive",
        "safety_false_positive",
        "gold_source_miss",
        "low_answer_relevancy",
    ]
    primary_tag = None
    for key in priority_order:
        if key in unique_tags:
            primary_tag = key
            break
    if primary_tag is None and unique_tags:
        primary_tag = unique_tags[0]

    return {
        "is_badcase": bool(unique_tags),
        "tags": unique_tags,
        "primary_tag": primary_tag,
        "severity": severity,
        "reason": "；".join(reasons[:3]) if reasons else "",
    }


def build_badcase_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(items)
    tag_counter: Counter[str] = Counter()
    severity_counter: Counter[str] = Counter()
    badcase_count = 0

    for item in items:
        badcase = item.get("badcase") or {}
        if not isinstance(badcase, dict):
            continue
        tags = badcase.get("tags") or []
        if isinstance(tags, list):
            tag_counter.update([_normalize_text(tag) for tag in tags if _normalize_text(tag)])
        severity = _normalize_text(badcase.get("severity")) or "none"
        severity_counter.update([severity])
        if bool(badcase.get("is_badcase")):
            badcase_count += 1

    badcase_rate = (badcase_count / total) if total > 0 else None
    return {
        "total_items": total,
        "badcase_count": badcase_count,
        "badcase_rate": round(badcase_rate, 6) if badcase_rate is not None else None,
        "severity_distribution": dict(severity_counter),
        "tag_distribution": dict(tag_counter),
        "top_tags": [{"tag": tag, "count": count} for tag, count in tag_counter.most_common(8)],
    }

