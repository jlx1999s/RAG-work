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


def _tokenize_text(text: str) -> set[str]:
    raw = _normalize_text(text).lower()
    if not raw:
        return set()
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", raw))


def _split_sentences(text: str, max_sentences: int = 12) -> List[str]:
    raw = _normalize_text(text)
    if not raw:
        return []
    parts = re.split(r"[。！？!?;\n]+", raw)
    sentences = [part.strip() for part in parts if part and part.strip()]
    if len(sentences) > max_sentences:
        return sentences[:max_sentences]
    return sentences


def _best_support(sentence: str, contexts: List[str]) -> Dict[str, Any]:
    sentence_tokens = _tokenize_text(sentence)
    if not sentence_tokens or not contexts:
        return {"best_index": None, "support_score": 0.0}
    best_idx = None
    best_score = 0.0
    for idx, context in enumerate(contexts):
        context_tokens = _tokenize_text(context)
        if not context_tokens:
            continue
        overlap = len(sentence_tokens & context_tokens)
        score = overlap / max(len(sentence_tokens), 1)
        if score > best_score:
            best_score = score
            best_idx = idx
    return {"best_index": best_idx, "support_score": best_score}


def build_item_generation_alignment(
    item: Dict[str, Any],
    *,
    support_threshold: float = 0.25,
) -> Dict[str, Any]:
    answer = _normalize_text(item.get("answer"))
    reference = _normalize_text(item.get("reference"))
    contexts = [str(c) for c in (item.get("contexts") or []) if _normalize_text(c)]
    answer_sentences = _split_sentences(answer)

    alignments: List[Dict[str, Any]] = []
    supported_count = 0
    for sentence in answer_sentences:
        support = _best_support(sentence, contexts)
        support_score = float(support.get("support_score") or 0.0)
        supported = support_score >= support_threshold
        if supported:
            supported_count += 1
        alignments.append(
            {
                "sentence": sentence,
                "best_context_index": support.get("best_index"),
                "support_score": round(support_score, 4),
                "supported": supported,
            }
        )

    ref_tokens = _tokenize_text(reference)
    answer_tokens = _tokenize_text(answer)
    overlap = len(ref_tokens & answer_tokens) if ref_tokens and answer_tokens else 0
    completeness = (
        overlap / max(len(ref_tokens), 1)
        if ref_tokens
        else None
    )
    citation_coverage = (
        supported_count / max(len(answer_sentences), 1)
        if answer_sentences
        else None
    )

    return {
        "alignments": alignments,
        "completeness": round(completeness, 6) if completeness is not None else None,
        "citation_coverage": round(citation_coverage, 6) if citation_coverage is not None else None,
        "supported_sentence_count": supported_count,
        "total_sentence_count": len(answer_sentences),
    }


def build_generation_lab_report(
    items: List[Dict[str, Any]],
    *,
    quality_summary: Dict[str, Any],
) -> Dict[str, Any]:
    completeness_values: List[float] = []
    citation_values: List[float] = []
    aligned_items = 0

    for item in items:
        alignment = build_item_generation_alignment(item)
        item["generation_alignment"] = alignment
        completeness = _safe_float(alignment.get("completeness"))
        if completeness is not None:
            completeness_values.append(completeness)
        citation = _safe_float(alignment.get("citation_coverage"))
        if citation is not None:
            citation_values.append(citation)
        if alignment.get("alignments"):
            aligned_items += 1

    return {
        "name": "Generation Lab",
        "enabled": True,
        "metrics": {
            "faithfulness": _safe_float(quality_summary.get("faithfulness")),
            "answer_relevancy": _safe_float(quality_summary.get("answer_relevancy")),
            "completeness": (
                sum(completeness_values) / len(completeness_values)
                if completeness_values
                else None
            ),
            "citation_coverage": (
                sum(citation_values) / len(citation_values)
                if citation_values
                else None
            ),
        },
        "scoring_layers": {
            "rule_based": {
                "completeness": (
                    sum(completeness_values) / len(completeness_values)
                    if completeness_values
                    else None
                ),
                "citation_coverage": (
                    sum(citation_values) / len(citation_values)
                    if citation_values
                    else None
                ),
            },
            "llm_as_judge": {
                "faithfulness": _safe_float(quality_summary.get("faithfulness")),
                "answer_relevancy": _safe_float(quality_summary.get("answer_relevancy")),
            },
        },
        "alignment_summary": {
            "aligned_items": aligned_items,
            "alignment_rate": _safe_divide(aligned_items, len(items)) if items else None,
        },
    }

