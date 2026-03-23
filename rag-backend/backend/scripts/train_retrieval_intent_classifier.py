#!/usr/bin/env python3
"""训练检索意图分类器（朴素贝叶斯）

支持企业实践中的 train/valid/test 切分、阈值建议与校准评估。
"""

import argparse
import datetime as dt
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agent.models.retrieval_classifier import tokenize_text


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_label(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if int(value) > 0 else 0
    text = _normalize_text(value).lower()
    if text in {"1", "true", "yes", "y", "need", "需要"}:
        return 1
    if text in {"0", "false", "no", "n", "noneed", "不需要"}:
        return 0
    raise ValueError(f"无法解析标签: {value}")


def _to_timestamp(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _normalize_text(value)
    if not text:
        return None
    try:
        # 兼容 ISO-8601
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.timestamp()
    except Exception:
        return None


def _extract_timestamp(payload: Dict[str, Any]) -> Optional[float]:
    for key in ("timestamp", "ts", "created_at", "createdAt", "date", "time"):
        if key not in payload:
            continue
        ts = _to_timestamp(payload.get(key))
        if ts is not None:
            return ts
    return None


def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"第{line_no}行不是对象")
        text = _normalize_text(payload.get("question") or payload.get("query") or payload.get("text"))
        if not text:
            raise ValueError(f"第{line_no}行缺少question/query/text")
        label_value = payload.get("need_retrieval")
        if label_value is None and "label" in payload:
            label_value = payload.get("label")
        if label_value is None:
            raise ValueError(f"第{line_no}行缺少need_retrieval/label")
        label = _to_label(label_value)
        rows.append(
            {
                "text": text,
                "label": label,
                "timestamp": _extract_timestamp(payload),
                "raw": payload,
            }
        )
    if not rows:
        raise ValueError("训练集为空")
    return rows


def _train_nb(
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
    for tokens, label in tokenized_samples:
        class_name = "pos" if label == 1 else "neg"
        class_counts[class_name] += 1
        for token in tokens:
            if token_freq[token] < min_token_freq:
                continue
            token_counts[token][class_name] += 1
            token_totals[class_name] += 1
    vocab_size = max(1, len(token_counts))
    return {
        "token_counts": token_counts,
        "class_counts": class_counts,
        "token_totals": token_totals,
        "vocab_size": vocab_size,
        "smoothing": float(smoothing),
    }


def _predict_prob(model: Dict[str, Any], text: str) -> float:
    tokens = tokenize_text(text)
    token_counts = model["token_counts"]
    class_counts = model["class_counts"]
    token_totals = model["token_totals"]
    vocab_size = max(1, int(model["vocab_size"]))
    smoothing = float(model["smoothing"])
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


def _binary_metrics_from_probabilities(prob_rows: List[Tuple[float, int]]) -> Dict[str, Any]:
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


def _evaluate(model: Dict[str, Any], samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    start = time.perf_counter()
    prob_rows: List[Tuple[float, int]] = []
    for sample in samples:
        prob = _predict_prob(model, sample["text"])
        prob_rows.append((prob, int(sample["label"])))
    elapsed_ms = (time.perf_counter() - start) * 1000
    metrics = _binary_metrics_from_probabilities(prob_rows)
    avg_latency_ms = (elapsed_ms / len(samples)) if samples else None
    metrics["avg_infer_latency_ms"] = round(avg_latency_ms, 6) if avg_latency_ms is not None else None
    metrics["prob_rows"] = prob_rows
    return metrics


def _compute_pr_auc(prob_rows: List[Tuple[float, int]]) -> Optional[float]:
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
    for r, p in curve:
        area += (r - prev_r) * ((p + prev_p) / 2.0)
        prev_r, prev_p = r, p
    if prev_r < 1.0:
        area += (1.0 - prev_r) * prev_p
    return round(area, 6)


def _compute_ece(prob_rows: List[Tuple[float, int]], num_bins: int = 10) -> Optional[float]:
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
    for i, bucket in enumerate(bins):
        if not bucket:
            curve.append(
                {
                    "bin_index": i,
                    "prob_min": round(i / num_bins, 4),
                    "prob_max": round((i + 1) / num_bins, 4),
                    "count": 0,
                    "avg_confidence": None,
                    "empirical_positive_rate": None,
                }
            )
            continue
        curve.append(
            {
                "bin_index": i,
                "prob_min": round(i / num_bins, 4),
                "prob_max": round((i + 1) / num_bins, 4),
                "count": len(bucket),
                "avg_confidence": round(sum(prob for prob, _ in bucket) / len(bucket), 6),
                "empirical_positive_rate": round(sum(label for _, label in bucket) / len(bucket), 6),
            }
        )
    return curve


def _suggest_thresholds(prob_rows: List[Tuple[float, int]]) -> Dict[str, Any]:
    if not prob_rows:
        return {"positive_threshold": 0.75, "negative_threshold": 0.25}
    # 企业实践中倾向更保守的正阈值，避免将不确定样本直接判为需要检索
    candidates = [i / 100.0 for i in range(60, 96, 2)]
    best_t = 0.75
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0
    for t in candidates:
        tp = fp = fn = 0
        for prob, label in prob_rows:
            pred = 1 if prob >= t else 0
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
            best_t = t
            best_precision = precision
            best_recall = recall

    # 负阈值使用与正阈值形成可解释不确定区间的策略
    # 保持明确的不确定区间，避免正负阈值过近导致边界抖动
    neg_t = max(0.05, min(0.45, best_t - 0.2))
    uncertain_count = sum(1 for prob, _ in prob_rows if neg_t < prob < best_t)
    return {
        "positive_threshold": round(best_t, 4),
        "negative_threshold": round(neg_t, 4),
        "expected_at_positive_threshold": {
            "f1": round(best_f1, 6),
            "precision": round(best_precision, 6),
            "recall": round(best_recall, 6),
        },
        "uncertain_rate_estimate": round(uncertain_count / len(prob_rows), 6) if prob_rows else None,
    }


def _split_samples(
    rows: List[Dict[str, Any]],
    *,
    seed: int,
    valid_ratio: float,
    test_ratio: float,
    strategy: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    prepared = list(rows)
    if strategy == "time":
        with_ts = [row for row in prepared if row.get("timestamp") is not None]
        without_ts = [row for row in prepared if row.get("timestamp") is None]
        if len(with_ts) >= max(3, int(len(prepared) * 0.6)):
            with_ts.sort(key=lambda x: float(x["timestamp"]))
            prepared = with_ts + without_ts
        else:
            strategy = "random"
    if strategy == "random":
        rng = random.Random(seed)
        rng.shuffle(prepared)

    total = len(prepared)
    test_size = int(total * test_ratio)
    valid_size = int(total * valid_ratio)
    if test_size <= 0 and total >= 10 and test_ratio > 0:
        test_size = 1
    if valid_size <= 0 and total >= 5 and valid_ratio > 0:
        valid_size = 1
    if test_size + valid_size >= total:
        # 至少留 1 条训练
        overflow = test_size + valid_size - (total - 1)
        if overflow > 0:
            if valid_size >= overflow:
                valid_size -= overflow
            else:
                overflow -= valid_size
                valid_size = 0
                test_size = max(0, test_size - overflow)

    test_rows = prepared[:test_size] if test_size > 0 else []
    valid_rows = prepared[test_size : test_size + valid_size] if valid_size > 0 else []
    train_rows = prepared[test_size + valid_size :]
    if not train_rows:
        train_rows = list(prepared)
        valid_rows = []
        test_rows = []
    return train_rows, valid_rows, test_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="训练JSONL路径")
    parser.add_argument("--output", required=True, help="输出模型JSON路径")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-strategy", choices=["random", "time"], default="random")
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--min-token-freq", type=int, default=2)
    parser.add_argument("--model-version", default="nb-v1")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = _load_dataset(input_path)

    train_rows, valid_rows, test_rows = _split_samples(
        rows,
        seed=args.seed,
        valid_ratio=max(0.0, args.valid_ratio),
        test_ratio=max(0.0, args.test_ratio),
        strategy=args.split_strategy,
    )

    model_data = _train_nb(
        train_rows,
        smoothing=args.smoothing,
        min_token_freq=max(1, args.min_token_freq),
    )
    train_metrics = _evaluate(model_data, train_rows)
    valid_metrics = _evaluate(model_data, valid_rows) if valid_rows else None
    test_metrics = _evaluate(model_data, test_rows) if test_rows else None

    # 以 valid 优先做阈值建议与校准，若无 valid 则退化到 train
    threshold_rows = (valid_metrics or train_metrics).get("prob_rows") or []
    threshold_suggestion = _suggest_thresholds(threshold_rows)
    calibration_curve = _build_calibration_curve(threshold_rows, num_bins=10)
    ece = _compute_ece(threshold_rows, num_bins=10)
    pr_auc = _compute_pr_auc(threshold_rows)

    for metrics in (train_metrics, valid_metrics, test_metrics):
        if isinstance(metrics, dict):
            metrics.pop("prob_rows", None)
            rows_for_auc = []
            if metrics is train_metrics:
                rows_for_auc = _evaluate(model_data, train_rows).get("prob_rows", [])
            elif metrics is valid_metrics and valid_rows:
                rows_for_auc = _evaluate(model_data, valid_rows).get("prob_rows", [])
            elif metrics is test_metrics and test_rows:
                rows_for_auc = _evaluate(model_data, test_rows).get("prob_rows", [])
            metrics["pr_auc"] = _compute_pr_auc(rows_for_auc) if rows_for_auc else None
            metrics["ece"] = _compute_ece(rows_for_auc, num_bins=10) if rows_for_auc else None
            metrics["recall_at_positive"] = metrics.get("recall")

    serializable_model = {
        "model_type": "naive_bayes_retrieval_intent",
        "model_version": args.model_version,
        "created_at": int(Path(args.input).stat().st_mtime),
        "smoothing": float(args.smoothing),
        "min_token_freq": int(max(1, args.min_token_freq)),
        "vocab_size": int(model_data["vocab_size"]),
        "class_counts": model_data["class_counts"],
        "token_totals": model_data["token_totals"],
        "token_counts": model_data["token_counts"],
        "metadata": {
            "input_path": str(input_path),
            "split_strategy": args.split_strategy,
            "seed": args.seed,
            "train_size": len(train_rows),
            "valid_size": len(valid_rows),
            "test_size": len(test_rows),
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "threshold_suggestion": threshold_suggestion,
            "calibration_curve": calibration_curve,
            "global_metrics": {
                "ece": ece,
                "pr_auc": pr_auc,
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serializable_model, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(output_path),
                "split_strategy": args.split_strategy,
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
                "test_metrics": test_metrics,
                "threshold_suggestion": threshold_suggestion,
                "global_metrics": {
                    "pr_auc": pr_auc,
                    "ece": ece,
                },
                "vocab_size": serializable_model["vocab_size"],
                "train_size": len(train_rows),
                "valid_size": len(valid_rows),
                "test_size": len(test_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
