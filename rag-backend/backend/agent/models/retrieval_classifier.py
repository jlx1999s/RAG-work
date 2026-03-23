import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def tokenize_text(text: str) -> List[str]:
    raw = _normalize_text(text).lower()
    if not raw:
        return []
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", raw)


class RetrievalIntentClassifier:
    """可训练的检索意图分类器（朴素贝叶斯推理）

    模型文件为JSON格式，可通过 scripts/train_retrieval_intent_classifier.py 生成。
    """

    def __init__(
        self,
        model_path: str,
        *,
        enabled: bool = True,
        positive_threshold: float = 0.75,
        negative_threshold: float = 0.25,
    ) -> None:
        self.model_path = model_path
        self.enabled = bool(enabled)
        self.positive_threshold = float(positive_threshold)
        self.negative_threshold = float(negative_threshold)
        if self.positive_threshold < self.negative_threshold:
            self.positive_threshold, self.negative_threshold = self.negative_threshold, self.positive_threshold

        self.ready = False
        self.error_message = ""
        self.model_version = "unloaded"
        self.loaded_at_ms: Optional[int] = None
        self.metadata: Dict[str, Any] = {}
        self.vocab_size = 0
        self.smoothing = 1.0
        self.class_counts = {"pos": 0, "neg": 0}
        self.token_totals = {"pos": 0, "neg": 0}
        self.token_counts: Dict[str, Dict[str, int]] = {}
        self._load_model()

    def _load_model(self) -> None:
        if not self.enabled:
            self.error_message = "classifier_disabled"
            return
        path = Path(self.model_path)
        if not path.exists():
            self.error_message = f"model_not_found:{self.model_path}"
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            token_counts = payload.get("token_counts") or {}
            class_counts = payload.get("class_counts") or {}
            token_totals = payload.get("token_totals") or {}
            if not isinstance(token_counts, dict):
                raise ValueError("token_counts_invalid")
            self.token_counts = {
                str(token): {
                    "pos": int((stats or {}).get("pos", 0)),
                    "neg": int((stats or {}).get("neg", 0)),
                }
                for token, stats in token_counts.items()
                if str(token).strip()
            }
            self.class_counts = {
                "pos": int(class_counts.get("pos", 0)),
                "neg": int(class_counts.get("neg", 0)),
            }
            self.token_totals = {
                "pos": int(token_totals.get("pos", 0)),
                "neg": int(token_totals.get("neg", 0)),
            }
            self.smoothing = float(payload.get("smoothing", 1.0))
            self.vocab_size = max(1, int(payload.get("vocab_size", len(self.token_counts) or 1)))
            self.model_version = str(payload.get("model_version") or payload.get("version") or "nb-v1")
            self.metadata = payload.get("metadata") or {}
            self.ready = True
            self.loaded_at_ms = int(time.time() * 1000)
            self.error_message = ""
        except Exception as exc:
            self.ready = False
            self.error_message = str(exc)

    def _log_prob(self, tokens: List[str], class_name: str) -> float:
        class_count = int(self.class_counts.get(class_name, 0))
        total_examples = max(1, int(self.class_counts.get("pos", 0)) + int(self.class_counts.get("neg", 0)))
        prior = math.log((class_count + 1.0) / (total_examples + 2.0))
        token_total = int(self.token_totals.get(class_name, 0))
        denominator = token_total + self.smoothing * self.vocab_size
        if denominator <= 0:
            denominator = 1.0
        log_prob = prior
        for token in tokens:
            stats = self.token_counts.get(token, {})
            token_count = int(stats.get(class_name, 0))
            likelihood = (token_count + self.smoothing) / denominator
            log_prob += math.log(max(likelihood, 1e-12))
        return float(log_prob)

    def predict(self, text: str) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "available": False,
                "decision": None,
                "reason": "classifier_disabled",
                "model_version": self.model_version,
            }
        if not self.ready:
            return {
                "available": False,
                "decision": None,
                "reason": self.error_message or "model_unavailable",
                "model_version": self.model_version,
            }
        tokens = tokenize_text(text)
        if not tokens:
            return {
                "available": True,
                "decision": False,
                "probability": 0.0,
                "confidence": 1.0,
                "band": "certain_no",
                "reason": "empty_input",
                "model_version": self.model_version,
                "thresholds": {
                    "positive": self.positive_threshold,
                    "negative": self.negative_threshold,
                },
            }

        log_pos = self._log_prob(tokens, "pos")
        log_neg = self._log_prob(tokens, "neg")
        logit = log_pos - log_neg
        try:
            prob_need_retrieval = 1.0 / (1.0 + math.exp(-logit))
        except OverflowError:
            prob_need_retrieval = 1.0 if logit > 0 else 0.0
        confidence = abs(prob_need_retrieval - 0.5) * 2.0
        decision: Optional[bool] = None
        band = "uncertain"
        if prob_need_retrieval >= self.positive_threshold:
            decision = True
            band = "certain_yes"
        elif prob_need_retrieval <= self.negative_threshold:
            decision = False
            band = "certain_no"

        return {
            "available": True,
            "decision": decision,
            "probability": round(prob_need_retrieval, 6),
            "confidence": round(confidence, 6),
            "band": band,
            "reason": (
                f"statistical_classifier:{band}, p_need_retrieval={prob_need_retrieval:.4f}, "
                f"thresholds=({self.negative_threshold:.2f},{self.positive_threshold:.2f})"
            ),
            "model_version": self.model_version,
            "model_path": self.model_path,
            "token_count": len(tokens),
            "thresholds": {
                "positive": self.positive_threshold,
                "negative": self.negative_threshold,
            },
            "metadata": self.metadata,
        }
