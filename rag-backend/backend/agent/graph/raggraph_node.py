from langgraph.runtime import Runtime
from ..states.raggraph_state import RAGGraphState
from ..contexts.raggraph_context import RAGContext
from ..models.raggraph_models import RetrievalMode, RetrievedDocument
from ..prompts.raggraph_prompt import (
    RAGGraphPrompts,
    RetrievalNeedDecision,
    SubquestionExpansion,
    ToolSkillDecision
)
from langchain_core.messages import AIMessage
from ...config.log import get_logger
from backend.agent.tools.exceptions import (
    ToolCallException,
    ToolExecutionError,
    ToolNotFoundError,
    ToolExecutionTimeoutError,
    ToolValidationError
)
from backend.agent.tools.audit import get_audit_logger
import asyncio
import copy
import hashlib
import json
import os
import time
import math
import re
import redis as redis_sync
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from urllib.parse import urlparse, unquote
from backend.config.oss import get_presigned_url_for_download
from backend.config.redis import get_redis_client

class RAGNodes:
    """RAG图节点实现类

    包含所有RAG图的节点实现和路由逻辑
    """

    def __init__(self, llm=None, embedding_model=None, milvus_storage=None, memory_store=None, checkpointer=None, lightrag_storage=None, tools=None):
        """初始化RAG节点

        Args:
            llm: 语言模型
            embedding_model: 嵌入模型
            milvus_storage: Milvus存储实例
            memory_store: 记忆存储实例
            checkpointer: 检查点存储实例
            lightrag_storage: LightRAG存储实例
            tools: 可用工具列表（如风险评估工具）
        """
        self.llm = llm
        self.embedding_model = embedding_model
        self.milvus_storage = milvus_storage
        self.memory_store = memory_store
        self.checkpointer = checkpointer
        self.lightrag_storage = lightrag_storage
        self.tools = tools or []  # 工具列表
        self.logger = get_logger(__name__)
        self.tool_timeout = 30.0  # 工具执行超时（30秒）
        self.executor = ThreadPoolExecutor(max_workers=5)  # 线程池用于超时控制
        self.rrf_k = float(os.getenv("RAG_RRF_K", "60"))
        self.graph_source_weight = float(os.getenv("RAG_GRAPH_SOURCE_WEIGHT", "1.1"))
        self.vector_source_weight = float(os.getenv("RAG_VECTOR_SOURCE_WEIGHT", "1.0"))
        self.mmr_lambda = float(os.getenv("RAG_MMR_LAMBDA", "0.75"))
        self.enable_semantic_rerank = os.getenv("RAG_ENABLE_SEMANTIC_RERANK", "true").lower() in {"true", "1", "yes", "on"}
        self.semantic_rerank_max_inputs = int(os.getenv("RAG_SEMANTIC_RERANK_MAX_INPUTS", "10"))
        self.semantic_rerank_total_count = 0
        self.semantic_rerank_fallback_count = 0
        self.semantic_rerank_metrics_enabled = os.getenv("RAG_SEMANTIC_RERANK_METRICS_ENABLED", "true").lower() in {"true", "1", "yes", "on"}
        self.metrics_redis_prefix = os.getenv("RAG_METRICS_REDIS_PREFIX", "rag:metrics")
        self.retrieval_cache_enabled = os.getenv("RAG_RETRIEVAL_CACHE_ENABLED", "true").lower() in {"true", "1", "yes", "on"}
        self.retrieval_cache_ttl_seconds = float(os.getenv("RAG_RETRIEVAL_CACHE_TTL_SECONDS", "300"))
        self.retrieval_cache_max_entries = int(os.getenv("RAG_RETRIEVAL_CACHE_MAX_ENTRIES", "512"))
        self.redis_retrieval_cache_enabled = os.getenv("RAG_REDIS_RETRIEVAL_CACHE_ENABLED", "true").lower() in {"true", "1", "yes", "on"}
        self.redis_retrieval_cache_prefix = os.getenv("RAG_REDIS_RETRIEVAL_CACHE_PREFIX", "rag:retrieval")
        self.graph_query_timeout_seconds = float(os.getenv("RAG_GRAPH_QUERY_TIMEOUT_SECONDS", "2.5"))
        self.enable_conditional_graph = os.getenv("RAG_ENABLE_CONDITIONAL_GRAPH", "true").lower() in {"true", "1", "yes", "on"}
        self.conditional_graph_min_vector_docs = int(os.getenv("RAG_CONDITIONAL_GRAPH_MIN_VECTOR_DOCS", "3"))
        self.conditional_graph_confidence_threshold = float(os.getenv("RAG_CONDITIONAL_GRAPH_CONFIDENCE_THRESHOLD", "0.22"))
        self.graph_latency_target_ms = float(os.getenv("RAG_GRAPH_LATENCY_TARGET_MS", "600"))
        self.graph_budget_min_docs = int(os.getenv("RAG_GRAPH_BUDGET_MIN_DOCS", "1"))
        self.graph_budget_max_docs = int(os.getenv("RAG_GRAPH_BUDGET_MAX_DOCS", "8"))
        self.graph_circuit_breaker_enabled = os.getenv("RAG_GRAPH_CIRCUIT_BREAKER_ENABLED", "true").lower() in {"true", "1", "yes", "on"}
        self.graph_circuit_fail_threshold = int(os.getenv("RAG_GRAPH_CIRCUIT_FAIL_THRESHOLD", "3"))
        self.graph_circuit_cooldown_seconds = float(os.getenv("RAG_GRAPH_CIRCUIT_COOLDOWN_SECONDS", "30"))
        self.graph_circuit_opened_until = 0.0
        self.graph_consecutive_failures = 0
        self.graph_latency_ema_ms = 0.0
        self.graph_latency_ema_alpha = float(os.getenv("RAG_GRAPH_LATENCY_EMA_ALPHA", "0.25"))
        self._vector_retrieval_cache = {}
        self._graph_retrieval_cache = {}
        self._sync_redis_client = None
        self.retrieval_cache_scope = str(getattr(self.lightrag_storage, "workspace", "") or "default")
        self.chunk_role_weights = {
            "contraindication": float(os.getenv("RAG_ROLE_WEIGHT_CONTRAINDICATION", "1.25")),
            "dosage": float(os.getenv("RAG_ROLE_WEIGHT_DOSAGE", "1.18")),
            "recommendation": float(os.getenv("RAG_ROLE_WEIGHT_RECOMMENDATION", "1.12")),
            "adverse_reaction": float(os.getenv("RAG_ROLE_WEIGHT_ADVERSE_REACTION", "1.08")),
            "indication": float(os.getenv("RAG_ROLE_WEIGHT_INDICATION", "1.05")),
            "general_medical": float(os.getenv("RAG_ROLE_WEIGHT_GENERAL", "1.0"))
        }
        self.key_clause_bonus = float(os.getenv("RAG_KEY_CLAUSE_BONUS", "0.015"))
        self.evidence_level_bonus = float(os.getenv("RAG_EVIDENCE_LEVEL_BONUS", "0.012"))
        self.assessment_skill_name = "health_risk_assessment"
        self.medical_safety_enabled = os.getenv("RAG_MEDICAL_SAFETY_ENABLED", "true").lower() in {"true", "1", "yes", "on"}
        self.medical_disclaimer_text = os.getenv(
            "RAG_MEDICAL_DISCLAIMER_TEXT",
            "免责声明：以下内容仅供健康科普与信息参考，不能替代执业医生的面诊诊断与个体化治疗建议。"
        ).strip()
        self.medical_emergency_text = os.getenv(
            "RAG_MEDICAL_EMERGENCY_TEXT",
            "如出现胸痛、呼吸困难、意识障碍、抽搐、持续高热不退或活动性出血等紧急情况，请立即拨打120或尽快前往急诊。"
        ).strip()
        self.medical_domain_markers = self._parse_env_list(
            os.getenv("RAG_MEDICAL_DOMAIN_MARKERS"),
            ["疾病", "症状", "诊断", "治疗", "用药", "副作用", "禁忌", "血压", "血糖", "糖尿病", "高血压", "发热", "发烧", "heart", "diabetes", "hypertension"]
        )
        self.medical_emergency_markers = self._parse_env_list(
            os.getenv("RAG_MEDICAL_EMERGENCY_MARKERS"),
            ["胸痛", "呼吸困难", "意识障碍", "昏迷", "抽搐", "大出血", "活动性出血", "持续高热", "高热不退", "stroke", "seizure", "unconscious", "急救", "急诊", "120"]
        )
        self.tool_pending_ttl_seconds = float(os.getenv("RAG_TOOL_PENDING_TTL_SECONDS", "180"))
        self.enable_subquestion_expansion = os.getenv("RAG_ENABLE_SUBQUESTION_EXPANSION", "true").lower() in {"true", "1", "yes", "on"}
        self.subquestion_max_count = int(os.getenv("RAG_SUBQUESTION_MAX_COUNT", "3"))
        self.subquestion_trigger_min_chars = int(os.getenv("RAG_SUBQUESTION_TRIGGER_MIN_CHARS", "20"))
        self.subquestion_complexity_min_score = float(os.getenv("RAG_SUBQUESTION_COMPLEXITY_MIN_SCORE", "2.0"))
        self.subquestion_intent_weight = float(os.getenv("RAG_SUBQUESTION_INTENT_WEIGHT", "1.0"))
        self.subquestion_coordination_weight = float(os.getenv("RAG_SUBQUESTION_COORDINATION_WEIGHT", "1.0"))
        self.subquestion_constraint_weight = float(os.getenv("RAG_SUBQUESTION_CONSTRAINT_WEIGHT", "1.0"))
        self.subquestion_intent_markers = self._parse_env_list(
            os.getenv("RAG_SUBQUESTION_INTENT_MARKERS"),
            ["什么", "哪些", "如何", "怎么", "多少", "是否", "能否", "为什么", "哪种", "区别", "对比", "步骤", "流程", "建议", "注意事项"]
        )
        self.subquestion_coordination_markers = self._parse_env_list(
            os.getenv("RAG_SUBQUESTION_COORDINATION_MARKERS"),
            ["以及", "并且", "同时", "分别", "和", "与", "并", "及"]
        )
        self.subquestion_constraint_patterns = self._parse_env_list(
            os.getenv("RAG_SUBQUESTION_CONSTRAINT_PATTERNS"),
            ["[0-9]{4}年", "近[一二三四五六七八九十0-9]+年", "儿童|孕妇|老年人|肝肾功能", "轻度|中度|重度", "空腹|饭前|饭后", "阶段|分期|级别", "禁忌|适应症|并发症|风险因素"]
        )
        self.retrieval_rule_force_yes_patterns = self._parse_env_list(
            os.getenv("RAG_RETRIEVAL_RULE_FORCE_YES_PATTERNS"),
            ["联系方式", "根据.*文档", "出处", "来源", "数据", "统计", "最新", "指南", "药物", "治疗", "副作用", "禁忌", "诊断", "检索", "知识库"]
        )
        self.retrieval_rule_force_no_patterns = self._parse_env_list(
            os.getenv("RAG_RETRIEVAL_RULE_FORCE_NO_PATTERNS"),
            ["^\\s*(你好|您好|hello|hi)\\s*$", "^\\s*(谢谢|感谢|再见|bye)\\s*$", "^\\s*你是谁\\s*[?？]?$", "^\\s*你能做什么\\s*[?？]?$"]
        )
        self.retrieval_classifier_yes_keywords = self._parse_env_list(
            os.getenv("RAG_RETRIEVAL_CLASSIFIER_YES_KEYWORDS"),
            ["谁", "何时", "哪里", "多少", "文档", "资料", "来源", "联系方式", "政策", "标准", "疾病", "药", "手术", "并发症", "指标", "数据库"]
        )
        self.retrieval_classifier_no_keywords = self._parse_env_list(
            os.getenv("RAG_RETRIEVAL_CLASSIFIER_NO_KEYWORDS"),
            ["写一首", "写一段", "脑暴", "创意", "段子", "翻译", "改写", "润色", "总结这段话", "数学题", "计算"]
        )
        self.retrieval_classifier_yes_threshold = float(os.getenv("RAG_RETRIEVAL_CLASSIFIER_YES_THRESHOLD", "2.0"))
        self.retrieval_classifier_margin = float(os.getenv("RAG_RETRIEVAL_CLASSIFIER_MARGIN", "1.0"))
        self.rule_library_version = os.getenv("RAG_RULE_LIBRARY_VERSION", "v1")
        self.rule_library_redis_enabled = os.getenv("RAG_RULE_LIBRARY_REDIS_ENABLED", "true").lower() in {"true", "1", "yes", "on"}
        self.rule_library_redis_prefix = os.getenv("RAG_RULE_LIBRARY_REDIS_PREFIX", "rag:rule_library")
        self.rule_library_cache_ttl_seconds = float(os.getenv("RAG_RULE_LIBRARY_CACHE_TTL_SECONDS", "60"))
        self.rule_library_env_json = os.getenv("RAG_RULE_LIBRARY_JSON", "")
        self._rule_library_cache = {"expires_at": 0.0, "payload": None}
        self.collection_policy_overrides = self._load_collection_policy_overrides(os.getenv("RAG_COLLECTION_POLICY_OVERRIDES", ""))
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.assessment_profiles = {
            "hypertension_risk_assessment": {
                "required_params": ["age", "systolic_bp", "diastolic_bp"],
                "display_name": "高血压风险评估",
                "trigger_keywords": ["高血压", "血压", "收缩压", "舒张压", "高压", "低压", "systolic", "diastolic", "systolic_bp", "diastolic_bp", "sbp", "dbp"]
            },
            "diabetes_risk_assessment": {
                "required_params": ["age", "bmi"],
                "display_name": "糖尿病风险评估",
                "trigger_keywords": ["糖尿病", "血糖", "bmi", "体重指数", "diabetes", "glucose"]
            }
        }

    def _parse_env_list(self, value: str | None, default: list[str]) -> list[str]:
        if value is None:
            return list(default)
        raw = str(value).strip()
        if not raw:
            return list(default)
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                pass
        return [part.strip() for part in raw.split(",") if part.strip()]

    def _load_collection_policy_overrides(self, raw_text: str) -> dict:
        if not raw_text:
            return {}
        try:
            payload = json.loads(raw_text)
            if not isinstance(payload, dict):
                return {}
            normalized = {}
            for key, value in payload.items():
                if isinstance(value, dict):
                    normalized[str(key)] = value
            return normalized
        except Exception as exc:
            self.logger.warning(f"解析RAG_COLLECTION_POLICY_OVERRIDES失败: {exc}")
            return {}

    def _policy_value(self, key: str, default):
        collection_overrides = self.collection_policy_overrides.get(self.retrieval_cache_scope, {})
        if key not in collection_overrides:
            return default
        return collection_overrides.get(key)

    def _policy_bool(self, key: str, default: bool) -> bool:
        value = self._policy_value(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes", "on"}
        return bool(value)

    def _policy_int(self, key: str, default: int) -> int:
        value = self._policy_value(key, default)
        try:
            return int(value)
        except Exception:
            return int(default)

    def _policy_float(self, key: str, default: float) -> float:
        value = self._policy_value(key, default)
        try:
            return float(value)
        except Exception:
            return float(default)

    def _policy_list(self, key: str, default: list[str]) -> list[str]:
        value = self._policy_value(key, default)
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return self._parse_env_list(value, default)
        return list(default)

    def _safe_int(self, value, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _safe_float(self, value, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _safe_bool(self, value, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes", "on"}
        if value is None:
            return default
        return bool(value)

    def _is_medical_text(self, text: str) -> bool:
        raw_text = str(text or "").lower()
        if not raw_text:
            return False
        return any(marker.lower() in raw_text for marker in self.medical_domain_markers)

    def _is_emergency_text(self, text: str) -> bool:
        raw_text = str(text or "").lower()
        if not raw_text:
            return False
        return any(marker.lower() in raw_text for marker in self.medical_emergency_markers)

    def _apply_medical_safety_notice(
        self,
        question: str,
        answer: str,
        *,
        force_medical: bool = False
    ) -> str:
        final_answer = str(answer or "").strip()
        if not final_answer:
            return final_answer
        if not self.medical_safety_enabled:
            return final_answer

        question_text = str(question or "").strip()
        answer_text = final_answer
        combined_text = f"{question_text}\n{answer_text}".strip()
        is_medical = force_medical or self._is_medical_text(combined_text)
        if not is_medical:
            return final_answer

        output_parts = [final_answer]
        if self.medical_disclaimer_text and self.medical_disclaimer_text not in answer_text:
            output_parts.append(self.medical_disclaimer_text)

        emergency_hit = self._is_emergency_text(question_text) or self._is_emergency_text(answer_text)
        if emergency_hit and self.medical_emergency_text and self.medical_emergency_text not in answer_text:
            output_parts.append(self.medical_emergency_text)

        return "\n\n".join(part for part in output_parts if part).strip()

    def _build_default_rule_library(self) -> dict:
        retrieval_rules = []
        for idx, pattern in enumerate(self.retrieval_rule_force_yes_patterns):
            retrieval_rules.append(
                {
                    "id": f"retrieval.force_yes.{idx + 1}",
                    "version": self.rule_library_version,
                    "scope": "global",
                    "stage": "retrieval",
                    "priority": 100 + idx,
                    "enabled": True,
                    "type": "regex",
                    "match": pattern,
                    "decision": True
                }
            )
        for idx, pattern in enumerate(self.retrieval_rule_force_no_patterns):
            retrieval_rules.append(
                {
                    "id": f"retrieval.force_no.{idx + 1}",
                    "version": self.rule_library_version,
                    "scope": "global",
                    "stage": "retrieval",
                    "priority": 200 + idx,
                    "enabled": True,
                    "type": "regex",
                    "match": pattern,
                    "decision": False
                }
            )
        retrieval_rules.append(
            {
                "id": "retrieval.force_no.math_expression",
                "version": self.rule_library_version,
                "scope": "global",
                "stage": "retrieval",
                "priority": 300,
                "enabled": True,
                "type": "math_expression",
                "decision": False
            }
        )
        return {
            "meta": {
                "version": self.rule_library_version,
                "source": "default"
            },
            "retrieval_rules": retrieval_rules,
            "subquestion_policy": {
                "id": "subquestion.default",
                "version": self.rule_library_version,
                "scope": "global",
                "complexity_min_score": self.subquestion_complexity_min_score,
                "intent_weight": self.subquestion_intent_weight,
                "coordination_weight": self.subquestion_coordination_weight,
                "constraint_weight": self.subquestion_constraint_weight,
                "trigger_min_chars": self.subquestion_trigger_min_chars,
                "max_count": self.subquestion_max_count
            }
        }

    def _normalize_rule_item(self, item: dict, fallback_scope: str) -> dict | None:
        if not isinstance(item, dict):
            return None
        rule_id = str(item.get("id", "")).strip()
        if not rule_id:
            return None
        stage = str(item.get("stage", "retrieval")).strip() or "retrieval"
        rule_type = str(item.get("type", "regex")).strip() or "regex"
        normalized = {
            "id": rule_id,
            "version": str(item.get("version", self.rule_library_version)).strip() or self.rule_library_version,
            "scope": str(item.get("scope", fallback_scope)).strip() or fallback_scope,
            "stage": stage,
            "priority": self._safe_int(item.get("priority", 999), 999),
            "enabled": self._safe_bool(item.get("enabled", True), True),
            "type": rule_type,
            "rollout_percent": self._safe_float(item.get("rollout_percent", 100.0), 100.0),
            "effective_from": self._safe_int(item.get("effective_from", 0), 0),
            "effective_to": self._safe_int(item.get("effective_to", 0), 0)
        }
        if "match" in item:
            normalized["match"] = str(item.get("match", ""))
        if "decision" in item:
            normalized["decision"] = bool(item.get("decision"))
        return normalized

    def _normalize_subquestion_policy(self, payload: dict, fallback_scope: str) -> dict:
        if not isinstance(payload, dict):
            payload = {}
        return {
            "id": str(payload.get("id", "subquestion.default")).strip() or "subquestion.default",
            "version": str(payload.get("version", self.rule_library_version)).strip() or self.rule_library_version,
            "scope": str(payload.get("scope", fallback_scope)).strip() or fallback_scope,
            "complexity_min_score": float(payload.get("complexity_min_score", self.subquestion_complexity_min_score)),
            "intent_weight": float(payload.get("intent_weight", self.subquestion_intent_weight)),
            "coordination_weight": float(payload.get("coordination_weight", self.subquestion_coordination_weight)),
            "constraint_weight": float(payload.get("constraint_weight", self.subquestion_constraint_weight)),
            "trigger_min_chars": self._safe_int(payload.get("trigger_min_chars", self.subquestion_trigger_min_chars), self.subquestion_trigger_min_chars),
            "max_count": self._safe_int(payload.get("max_count", self.subquestion_max_count), self.subquestion_max_count)
        }

    def _merge_rule_library(self, base_payload: dict, override_payload: dict, source_name: str) -> dict:
        merged = copy.deepcopy(base_payload) if isinstance(base_payload, dict) else self._build_default_rule_library()
        if not isinstance(override_payload, dict):
            return merged
        if isinstance(override_payload.get("meta"), dict):
            merged["meta"] = {**(merged.get("meta", {})), **override_payload.get("meta", {})}
            merged["meta"]["source"] = source_name
        incoming_rules = override_payload.get("retrieval_rules")
        if isinstance(incoming_rules, list):
            current_rules = {
                rule.get("id"): rule
                for rule in merged.get("retrieval_rules", [])
                if isinstance(rule, dict) and rule.get("id")
            }
            for item in incoming_rules:
                normalized_item = self._normalize_rule_item(item, fallback_scope="global")
                if normalized_item is not None:
                    current_rules[normalized_item["id"]] = normalized_item
            merged["retrieval_rules"] = list(current_rules.values())
        if isinstance(override_payload.get("subquestion_policy"), dict):
            merged["subquestion_policy"] = self._normalize_subquestion_policy(
                override_payload["subquestion_policy"],
                fallback_scope="global"
            )
        if "meta" not in merged:
            merged["meta"] = {}
        merged["meta"]["source"] = source_name
        return merged

    def _is_rule_in_effective_window(self, rule: dict, now_ts: int) -> tuple[bool, str]:
        start_ts = self._safe_int(rule.get("effective_from", 0), 0)
        end_ts = self._safe_int(rule.get("effective_to", 0), 0)
        if start_ts > 0 and now_ts < start_ts:
            return False, "not_started"
        if end_ts > 0 and now_ts > end_ts:
            return False, "expired"
        return True, ""

    def _is_rule_in_rollout(self, rule: dict, query_text: str) -> tuple[bool, str]:
        rollout_percent = self._safe_float(rule.get("rollout_percent", 100.0), 100.0)
        rollout_percent = max(0.0, min(100.0, rollout_percent))
        if rollout_percent >= 100.0:
            return True, ""
        seed = f"{rule.get('id', '')}:{query_text or ''}"
        bucket = int(hashlib.md5(seed.encode("utf-8")).hexdigest()[:8], 16) % 100
        if float(bucket) < rollout_percent:
            return True, ""
        return False, f"rollout_filtered({bucket}>={rollout_percent})"

    def _audit_rule_conflicts(self, rules: list[dict]) -> list[dict]:
        conflict_items = []
        grouped = {}
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            if str(rule.get("stage", "")) != "retrieval":
                continue
            if "decision" not in rule:
                continue
            key = (self._safe_int(rule.get("priority", 999), 999), str(rule.get("scope", "global")))
            grouped.setdefault(key, []).append(rule)
        for (priority, scope), bucket in grouped.items():
            decisions = {bool(item.get("decision")) for item in bucket}
            if len(bucket) > 1 and len(decisions) > 1:
                conflict_items.append(
                    {
                        "priority": priority,
                        "scope": scope,
                        "rule_ids": [str(item.get("id", "")) for item in bucket]
                    }
                )
        if conflict_items:
            self.logger.warning(f"规则库检测到优先级冲突: {conflict_items}")
        return conflict_items

    def _load_rule_library_from_redis(self) -> dict:
        if not self.rule_library_redis_enabled:
            return {}
        try:
            redis_client = self._get_sync_redis_client()
            global_key = f"{self.rule_library_redis_prefix}:global"
            raw_global = redis_client.get(global_key)
            merged_payload = {}
            if raw_global:
                merged_payload = self._merge_rule_library(
                    merged_payload,
                    json.loads(raw_global),
                    source_name="redis_global"
                )
            if self.retrieval_cache_scope:
                scope_key = f"{self.rule_library_redis_prefix}:collection:{self.retrieval_cache_scope}"
                raw_scope = redis_client.get(scope_key)
                if raw_scope:
                    merged_payload = self._merge_rule_library(
                        merged_payload,
                        json.loads(raw_scope),
                        source_name="redis_collection"
                    )
            return merged_payload
        except Exception as exc:
            self.logger.warning(f"读取规则库Redis配置失败: {exc}")
            return {}

    def _get_rule_library(self, force_refresh: bool = False) -> dict:
        now_ts = time.time()
        if (not force_refresh) and self._rule_library_cache.get("payload") is not None and now_ts < float(self._rule_library_cache.get("expires_at", 0.0)):
            return copy.deepcopy(self._rule_library_cache["payload"])
        payload = self._build_default_rule_library()
        if self.rule_library_env_json:
            try:
                env_payload = json.loads(self.rule_library_env_json)
                payload = self._merge_rule_library(payload, env_payload, source_name="env")
            except Exception as exc:
                self.logger.warning(f"解析RAG_RULE_LIBRARY_JSON失败: {exc}")
        redis_payload = self._load_rule_library_from_redis()
        if redis_payload:
            payload = self._merge_rule_library(payload, redis_payload, source_name="redis")
        collection_overrides = self.collection_policy_overrides.get(self.retrieval_cache_scope, {})
        if isinstance(collection_overrides, dict) and "rule_library" in collection_overrides:
            payload = self._merge_rule_library(payload, collection_overrides.get("rule_library"), source_name="collection_override")
        if not isinstance(payload.get("retrieval_rules"), list):
            payload["retrieval_rules"] = []
        payload["retrieval_rules"] = sorted(
            [rule for rule in payload.get("retrieval_rules", []) if isinstance(rule, dict)],
            key=lambda item: (self._safe_int(item.get("priority", 999), 999), str(item.get("id", "")))
        )
        payload["subquestion_policy"] = self._normalize_subquestion_policy(
            payload.get("subquestion_policy", {}),
            fallback_scope=self.retrieval_cache_scope
        )
        conflicts = self._audit_rule_conflicts(payload["retrieval_rules"])
        payload["meta"] = payload.get("meta", {})
        payload["meta"]["conflict_count"] = len(conflicts)
        payload["meta"]["conflicts"] = conflicts
        self._rule_library_cache = {
            "expires_at": now_ts + max(self.rule_library_cache_ttl_seconds, 1.0),
            "payload": copy.deepcopy(payload)
        }
        return payload

    def _extract_latest_message(self, state: RAGGraphState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return ""
        latest = messages[-1]
        return latest.content if hasattr(latest, "content") else str(latest)

    def _rule_first_retrieval_decision(self, question: str, library: dict | None = None) -> dict:
        normalized = (question or "").strip()
        if not normalized:
            return {
                "decision": False,
                "reason": "问题为空，跳过检索",
                "hit_rule_id": "retrieval.force_no.empty",
                "rule_version": self.rule_library_version,
                "rule_scope": "global",
                "matched_pattern": "",
                "decision_path": ["empty_question"]
            }
        library = library if isinstance(library, dict) else self._get_rule_library()
        retrieval_rules = library.get("retrieval_rules", [])
        now_ts = int(time.time())
        decision_path = []
        for rule in retrieval_rules:
            if not self._safe_bool(rule.get("enabled", True), True):
                decision_path.append(f"skip_disabled:{rule.get('id', '')}")
                continue
            if str(rule.get("stage", "retrieval")) != "retrieval":
                decision_path.append(f"skip_stage:{rule.get('id', '')}")
                continue
            in_window, window_reason = self._is_rule_in_effective_window(rule, now_ts)
            if not in_window:
                decision_path.append(f"skip_window:{rule.get('id', '')}:{window_reason}")
                continue
            in_rollout, rollout_reason = self._is_rule_in_rollout(rule, normalized)
            if not in_rollout:
                decision_path.append(f"skip_rollout:{rule.get('id', '')}:{rollout_reason}")
                continue
            rule_type = str(rule.get("type", "regex"))
            if rule_type == "regex":
                pattern = str(rule.get("match", "")).strip()
                if pattern and re.search(pattern, normalized, re.IGNORECASE):
                    decision = bool(rule.get("decision"))
                    reason_prefix = "强制检索" if decision else "免检索"
                    return {
                        "decision": decision,
                        "reason": f"规则库命中{reason_prefix}: {pattern}",
                        "hit_rule_id": rule.get("id"),
                        "rule_version": rule.get("version", self.rule_library_version),
                        "rule_scope": rule.get("scope", "global"),
                        "matched_pattern": pattern,
                        "decision_path": decision_path + [f"hit:{rule.get('id', '')}"]
                    }
                decision_path.append(f"miss:{rule.get('id', '')}")
            elif rule_type == "math_expression":
                if re.fullmatch(r"[0-9\.\+\-\*/\(\)\s=]+", normalized):
                    decision = bool(rule.get("decision"))
                    reason_prefix = "强制检索" if decision else "免检索"
                    return {
                        "decision": decision,
                        "reason": f"规则库命中{reason_prefix}: 纯计算表达式",
                        "hit_rule_id": rule.get("id"),
                        "rule_version": rule.get("version", self.rule_library_version),
                        "rule_scope": rule.get("scope", "global"),
                        "matched_pattern": "math_expression",
                        "decision_path": decision_path + [f"hit:{rule.get('id', '')}"]
                    }
                decision_path.append(f"miss:{rule.get('id', '')}")
            else:
                decision_path.append(f"skip_type:{rule.get('id', '')}:{rule_type}")
        return {
            "decision": None,
            "reason": "",
            "hit_rule_id": "",
            "rule_version": "",
            "rule_scope": "",
            "matched_pattern": "",
            "decision_path": decision_path
        }

    def _lightweight_retrieval_classifier(self, question: str) -> dict:
        normalized = (question or "").strip()
        yes_keywords = self._policy_list("retrieval_classifier_yes_keywords", self.retrieval_classifier_yes_keywords)
        no_keywords = self._policy_list("retrieval_classifier_no_keywords", self.retrieval_classifier_no_keywords)
        yes_threshold = self._policy_float("retrieval_classifier_yes_threshold", self.retrieval_classifier_yes_threshold)
        margin_threshold = self._policy_float("retrieval_classifier_margin", self.retrieval_classifier_margin)
        yes_hit = sum(1 for keyword in yes_keywords if keyword and keyword in normalized)
        no_hit = sum(1 for keyword in no_keywords if keyword and keyword in normalized)
        question_signal = sum(1 for signal in ["?", "？", "谁", "何时", "哪里", "多少", "哪", "怎么", "如何"] if signal in normalized)
        yes_score = float(yes_hit + 0.5 * question_signal)
        no_score = float(no_hit)
        margin = yes_score - no_score
        decision = None
        reason = "轻量分类器不确定"
        if yes_score >= yes_threshold and margin >= margin_threshold:
            decision = True
            reason = f"轻量分类器判定需要检索: yes_score={yes_score:.2f}, no_score={no_score:.2f}"
        elif no_score > yes_score and (no_score - yes_score) >= margin_threshold:
            decision = False
            reason = f"轻量分类器判定无需检索: yes_score={yes_score:.2f}, no_score={no_score:.2f}"
        return {
            "decision": decision,
            "reason": reason,
            "yes_score": round(yes_score, 4),
            "no_score": round(no_score, 4),
            "margin": round(margin, 4)
        }

    def _compute_subquestion_complexity(self, question: str) -> dict:
        normalized = (question or "").strip()
        library = self._get_rule_library()
        policy = library.get("subquestion_policy", {})
        intent_markers = self._policy_list("subquestion_intent_markers", self.subquestion_intent_markers)
        coordination_markers = self._policy_list("subquestion_coordination_markers", self.subquestion_coordination_markers)
        constraint_patterns = self._policy_list("subquestion_constraint_patterns", self.subquestion_constraint_patterns)
        intent_weight = self._policy_float("subquestion_intent_weight", float(policy.get("intent_weight", self.subquestion_intent_weight)))
        coordination_weight = self._policy_float("subquestion_coordination_weight", float(policy.get("coordination_weight", self.subquestion_coordination_weight)))
        constraint_weight = self._policy_float("subquestion_constraint_weight", float(policy.get("constraint_weight", self.subquestion_constraint_weight)))
        complexity_min_score = self._policy_float("subquestion_complexity_min_score", float(policy.get("complexity_min_score", self.subquestion_complexity_min_score)))
        intent_count = sum(1 for marker in intent_markers if marker and marker in normalized)
        coordination_count = sum(normalized.count(marker) for marker in coordination_markers if marker)
        constraint_count = 0
        for pattern in constraint_patterns:
            if pattern and re.search(pattern, normalized, re.IGNORECASE):
                constraint_count += 1
        score = intent_count * intent_weight + coordination_count * coordination_weight + constraint_count * constraint_weight
        return {
            "intent_count": intent_count,
            "coordination_count": coordination_count,
            "constraint_count": constraint_count,
            "score": round(score, 4),
            "min_score": round(complexity_min_score, 4),
            "triggered": score >= complexity_min_score,
            "policy_id": str(policy.get("id", "subquestion.default")),
            "policy_version": str(policy.get("version", self.rule_library_version)),
            "policy_scope": str(policy.get("scope", self.retrieval_cache_scope))
        }

    def _resolve_embedding_model_tag(self) -> str:
        env_model = os.getenv("VECTOR_DASHSCOPE_EMBEDDING_MODEL", "").strip()
        if env_model:
            return env_model
        model_name = getattr(self.embedding_model, "model", None) or getattr(self.embedding_model, "model_name", None)
        if model_name:
            return str(model_name)
        return type(self.embedding_model).__name__ if self.embedding_model is not None else "unknown_embedding"

    def _emit_semantic_rerank_metrics(self, rerank_stats: dict) -> dict:
        result = {"emitted": False, "error": "", "labels": {}}
        if not self.semantic_rerank_metrics_enabled:
            result["error"] = "metrics_disabled"
            return result
        if not rerank_stats.get("attempted"):
            result["error"] = "not_attempted"
            return result
        model_name = self._resolve_embedding_model_tag()
        collection_id = self.retrieval_cache_scope
        now_ts = int(time.time())
        labels = {"model": model_name, "collection": collection_id}
        result["labels"] = labels
        try:
            redis_client = self._get_sync_redis_client()
            windows = [
                ("1m", now_ts // 60),
                ("1h", now_ts // 3600)
            ]
            for window_name, bucket in windows:
                redis_key = (
                    f"{self.metrics_redis_prefix}:semantic_rerank:"
                    f"window={window_name}:bucket={bucket}:collection={collection_id}:model={model_name}"
                )
                redis_client.hincrby(redis_key, "total", 1)
                if rerank_stats.get("fallback"):
                    redis_client.hincrby(redis_key, "fallback", 1)
                redis_client.expire(redis_key, 7 * 24 * 3600)
            result["emitted"] = True
            return result
        except Exception as exc:
            result["error"] = str(exc)
            self.logger.warning(f"语义重排指标写入失败: {exc}")
            return result
    
    def _execute_tool_with_timeout(self, tool, tool_args, timeout=None):
        """带超时控制的工具执行
        
        Args:
            tool: 工具实例
            tool_args: 工具参数
            timeout: 超时时间（秒），默认使用 self.tool_timeout
        
        Returns:
            工具执行结果
        
        Raises:
            ToolExecutionTimeoutError: 工具执行超时
        """
        timeout = timeout or self.tool_timeout
        
        try:
            # 使用线程池执行工具，并设置超时
            future = self.executor.submit(tool.invoke, tool_args)
            result = future.result(timeout=timeout)
            return result
        except FutureTimeoutError:
            # 超时异常
            raise ToolExecutionTimeoutError(
                tool_name=tool.name,
                timeout=timeout,
                details=f"工具 '{tool.name}' 执行超过 {timeout} 秒"
            )

    # ==================== 节点实现 ====================

    def start_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        """开始节点

        RAG处理流程的入口节点，主要用于日志记录和状态确认。
        状态已在create_initial_rag_state函数中完成初始化。

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态
        """
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: START - 开始处理")

        return state

    def _clone_retrieved_docs(self, docs: list[RetrievedDocument]) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                page_content=doc.page_content,
                metadata=dict(doc.metadata or {})
            )
            for doc in (docs or [])
        ]

    def _normalize_query_for_cache(self, query_text: str) -> str:
        return re.sub(r"\s+", " ", (query_text or "").strip().lower())

    def _build_retrieval_cache_key(self, prefix: str, query_text: str, subquestions: list[str], max_docs: int, extra: str = "") -> str:
        normalized_query = self._normalize_query_for_cache(query_text)
        return f"{prefix}|scope={self.retrieval_cache_scope}|q={normalized_query}|k={max_docs}|x={extra}"

    def _extract_contact_candidates(self, retrieved_docs: list[RetrievedDocument]):
        candidates = []
        seen = set()
        for index, doc in enumerate(retrieved_docs or [], start=1):
            text = (doc.page_content or "").strip()
            if not text:
                continue
            emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
            raw_phones = re.findall(r"(?:\+?86[- ]?)?(?:1[3-9]\d[- ]?\d{4}[- ]?\d{4})", text)
            phones = []
            for raw_phone in raw_phones:
                cleaned_phone = re.sub(r"\D", "", raw_phone)
                if cleaned_phone.startswith("86") and len(cleaned_phone) > 11:
                    cleaned_phone = cleaned_phone[-11:]
                if re.fullmatch(r"1[3-9]\d{9}", cleaned_phone):
                    phones.append(cleaned_phone)
            if not emails and not phones:
                continue
            document_name = doc.metadata.get("document_name", f"文档{index}")
            for value in phones + emails:
                if value in seen:
                    continue
                seen.add(value)
                candidates.append({
                    "value": value,
                    "kind": "phone" if value.isdigit() and len(value) == 11 else "email",
                    "doc_index": index,
                    "document_name": document_name
                })
        return candidates

    def _repair_contact_answer_if_needed(self, question: str, answer: str, retrieved_docs: list[RetrievedDocument]) -> str:
        normalized_answer = (answer or "").strip()
        if "联系方式" not in (question or ""):
            return normalized_answer
        if not re.search(r"未.*上下文.*找到答案|未.*找到答案|无法.*给定上下文|没有找到相关信息", normalized_answer):
            return normalized_answer
        candidates = self._extract_contact_candidates(retrieved_docs)
        if not candidates:
            return normalized_answer
        phones = [item for item in candidates if item["kind"] == "phone"]
        emails = [item for item in candidates if item["kind"] == "email"]
        parts = []
        if phones:
            parts.append("电话 " + "、".join(item["value"] for item in phones[:3]))
        if emails:
            parts.append("邮箱 " + "、".join(item["value"] for item in emails[:3]))
        ref_ids = []
        for item in candidates:
            ref_id = item["doc_index"]
            if ref_id not in ref_ids:
                ref_ids.append(ref_id)
        refs = "".join(f"[{idx}]" for idx in ref_ids[:6])
        return f"根据检索到的上下文，联系方式为：{'；'.join(parts)}。{refs}"

    def _extract_memory_guidance(self, state: RAGGraphState) -> str:
        messages = state.get("messages", []) or []
        for msg in messages:
            role = getattr(msg, "type", "")
            content = msg.content if hasattr(msg, "content") else str(msg)
            if role == "system" and isinstance(content, str) and content.startswith("请在回答时优先遵循以下用户记忆。"):
                return content.strip()
        return ""

    def _get_cached_item(self, cache_store: dict, cache_key: str):
        if not self.retrieval_cache_enabled:
            return None
        payload = cache_store.get(cache_key)
        if not payload:
            return None
        expires_at, value = payload
        if time.time() >= expires_at:
            cache_store.pop(cache_key, None)
            return None
        return value

    def _set_cached_item(self, cache_store: dict, cache_key: str, value) -> None:
        if not self.retrieval_cache_enabled:
            return
        if len(cache_store) >= self.retrieval_cache_max_entries:
            oldest_key = min(cache_store.keys(), key=lambda key: cache_store[key][0])
            cache_store.pop(oldest_key, None)
        expires_at = time.time() + self.retrieval_cache_ttl_seconds
        cache_store[cache_key] = (expires_at, value)

    def _is_graph_circuit_open(self) -> bool:
        if not self.graph_circuit_breaker_enabled:
            return False
        return time.time() < self.graph_circuit_opened_until

    def _record_graph_success(self, duration_ms: float) -> None:
        self.graph_consecutive_failures = 0
        if duration_ms > 0:
            if self.graph_latency_ema_ms <= 0:
                self.graph_latency_ema_ms = duration_ms
            else:
                alpha = min(max(self.graph_latency_ema_alpha, 0.05), 0.9)
                self.graph_latency_ema_ms = alpha * duration_ms + (1.0 - alpha) * self.graph_latency_ema_ms

    def _record_graph_failure(self) -> None:
        if not self.graph_circuit_breaker_enabled:
            return
        self.graph_consecutive_failures += 1
        if self.graph_consecutive_failures >= max(1, self.graph_circuit_fail_threshold):
            self.graph_circuit_opened_until = time.time() + max(1.0, self.graph_circuit_cooldown_seconds)

    def _resolve_graph_budget_docs(self, base_max_docs: int) -> int:
        base = max(1, int(base_max_docs))
        min_docs = max(1, self.graph_budget_min_docs)
        max_docs = max(min_docs, self.graph_budget_max_docs)
        if self.graph_latency_ema_ms <= 0:
            return min(max(base, min_docs), max_docs)
        if self.graph_latency_ema_ms > self.graph_latency_target_ms * 1.5:
            target = max(min_docs, base - 1)
        elif self.graph_latency_ema_ms < self.graph_latency_target_ms * 0.7:
            target = min(max_docs, base + 1)
        else:
            target = base
        return min(max(target, min_docs), max_docs)

    def _serialize_retrieved_docs(self, docs: list[RetrievedDocument]) -> list[dict]:
        serialized = []
        for doc in (docs or []):
            serialized.append({
                "page_content": doc.page_content,
                "metadata": dict(doc.metadata or {})
            })
        return serialized

    def _deserialize_retrieved_docs(self, docs_payload: list[dict]) -> list[RetrievedDocument]:
        restored = []
        for item in (docs_payload or []):
            restored.append(
                RetrievedDocument(
                    page_content=item.get("page_content", ""),
                    metadata=dict(item.get("metadata") or {})
                )
            )
        return restored

    def _serialize_vector_cache_payload(self, payload) -> dict:
        retrieved_docs, vector_docs = payload
        return {
            "retrieved_docs": self._serialize_retrieved_docs(retrieved_docs),
            "vector_docs": self._serialize_retrieved_docs(vector_docs)
        }

    def _deserialize_vector_cache_payload(self, payload: dict):
        return (
            self._deserialize_retrieved_docs(payload.get("retrieved_docs") or []),
            self._deserialize_retrieved_docs(payload.get("vector_docs") or [])
        )

    async def _get_redis_retrieval_cache(self, cache_key: str):
        if not (self.retrieval_cache_enabled and self.redis_retrieval_cache_enabled):
            return None
        try:
            redis_client = await get_redis_client()
            redis_key = f"{self.redis_retrieval_cache_prefix}:{cache_key}"
            cached_text = await redis_client.get(redis_key)
            if not cached_text:
                return None
            return json.loads(cached_text)
        except Exception as exc:
            self.logger.warning(f"读取Redis检索缓存失败: {exc}")
            return None

    async def _set_redis_retrieval_cache(self, cache_key: str, payload: dict) -> None:
        if not (self.retrieval_cache_enabled and self.redis_retrieval_cache_enabled):
            return
        try:
            redis_client = await get_redis_client()
            redis_key = f"{self.redis_retrieval_cache_prefix}:{cache_key}"
            ttl_seconds = max(1, int(self.retrieval_cache_ttl_seconds))
            await redis_client.set(redis_key, json.dumps(payload, ensure_ascii=False), ex=ttl_seconds)
        except Exception as exc:
            self.logger.warning(f"写入Redis检索缓存失败: {exc}")

    def _get_sync_redis_client(self):
        if self._sync_redis_client is not None:
            return self._sync_redis_client
        self._sync_redis_client = redis_sync.Redis(
            host=os.getenv("REDIS_HOST", "127.0.0.1"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
            retry_on_timeout=True
        )
        return self._sync_redis_client

    def _get_redis_retrieval_cache_sync(self, cache_key: str):
        if not (self.retrieval_cache_enabled and self.redis_retrieval_cache_enabled):
            return None
        try:
            redis_client = self._get_sync_redis_client()
            redis_key = f"{self.redis_retrieval_cache_prefix}:{cache_key}"
            cached_text = redis_client.get(redis_key)
            if not cached_text:
                return None
            return json.loads(cached_text)
        except Exception as exc:
            self.logger.warning(f"同步读取Redis检索缓存失败: {exc}")
            return None

    def _set_redis_retrieval_cache_sync(self, cache_key: str, payload: dict) -> None:
        if not (self.retrieval_cache_enabled and self.redis_retrieval_cache_enabled):
            return
        try:
            redis_client = self._get_sync_redis_client()
            redis_key = f"{self.redis_retrieval_cache_prefix}:{cache_key}"
            ttl_seconds = max(1, int(self.retrieval_cache_ttl_seconds))
            redis_client.set(redis_key, json.dumps(payload, ensure_ascii=False), ex=ttl_seconds)
        except Exception as exc:
            self.logger.warning(f"同步写入Redis检索缓存失败: {exc}")

    async def _get_cached_item_async(self, cache_store: dict, cache_key: str, payload_parser):
        local_cached = self._get_cached_item(cache_store, cache_key)
        if local_cached is not None:
            return local_cached
        redis_payload = await self._get_redis_retrieval_cache(cache_key)
        if redis_payload is None:
            return None
        parsed = payload_parser(redis_payload)
        self._set_cached_item(cache_store, cache_key, parsed)
        return parsed

    async def _set_cached_item_async(self, cache_store: dict, cache_key: str, value, payload_builder) -> None:
        self._set_cached_item(cache_store, cache_key, value)
        payload = payload_builder(value)
        await self._set_redis_retrieval_cache(cache_key, payload)

    def _extract_hypertension_slots(self, question: str) -> dict:
        slots = {}
        age_match = re.search(r"(?:年龄|age)\s*[:：]?\s*(\d{1,3})", question, re.IGNORECASE)
        if not age_match:
            age_match = re.search(r"(\d{1,3})\s*岁", question)
        if age_match:
            slots["age"] = int(age_match.group(1))
        bp_pair_match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", question)
        if bp_pair_match:
            slots["systolic_bp"] = int(bp_pair_match.group(1))
            slots["diastolic_bp"] = int(bp_pair_match.group(2))
        else:
            systolic_match = re.search(r"(?:收缩压|高压|systolic(?:_bp)?|sbp)\s*(?:[:：=]|是|为)?\s*(\d{2,3})", question, re.IGNORECASE)
            diastolic_match = re.search(r"(?:舒张压|低压|diastolic(?:_bp)?|dbp)\s*(?:[:：=]|是|为)?\s*(\d{2,3})", question, re.IGNORECASE)
            if systolic_match:
                slots["systolic_bp"] = int(systolic_match.group(1))
            if diastolic_match:
                slots["diastolic_bp"] = int(diastolic_match.group(1))
        return slots

    def _extract_message_content(self, message) -> str:
        if hasattr(message, "content"):
            return str(getattr(message, "content", "") or "")
        if isinstance(message, dict):
            return str(message.get("content", "") or "")
        return str(message or "")

    def _is_assistant_message(self, message) -> bool:
        if hasattr(message, "type"):
            return str(getattr(message, "type", "")).lower() in {"ai", "assistant"}
        if isinstance(message, dict):
            return str(message.get("role", "")).lower() in {"assistant", "ai"}
        return False

    def _resolve_pending_assessment_tool(self, messages: list) -> str:
        if not messages:
            return ""
        for message in reversed(messages):
            if not self._is_assistant_message(message):
                continue
            content = self._extract_message_content(message)
            if "还需要补充以下参数" not in content:
                return ""
            for tool_name, profile in self.assessment_profiles.items():
                display_name = profile.get("display_name", "")
                if display_name and display_name in content:
                    return tool_name
                if tool_name in content:
                    return tool_name
            if "高血压" in content or "血压" in content:
                return "hypertension_risk_assessment" if "hypertension_risk_assessment" in self.tool_map else ""
            if "糖尿病" in content or "血糖" in content or "bmi" in content.lower():
                return "diabetes_risk_assessment" if "diabetes_risk_assessment" in self.tool_map else ""
            return ""
        return ""

    def _is_pending_followup_for_tool(self, tool_name: str, question: str) -> bool:
        if not tool_name:
            return False
        question_text = str(question or "").strip().lower()
        if not question_text:
            return False
        extracted_slots = self._extract_required_slots(tool_name, question_text)
        if extracted_slots:
            return True
        profile = self.assessment_profiles.get(tool_name, {})
        trigger_keywords = [str(item).lower() for item in profile.get("trigger_keywords", [])]
        return any(keyword and keyword in question_text for keyword in trigger_keywords)

    def _clear_pending_tool_state(self, state: RAGGraphState) -> None:
        state["pending_tool_name"] = ""
        state["pending_tool_deadline_ms"] = 0

    def _set_pending_tool_state(self, state: RAGGraphState, tool_name: str) -> None:
        ttl_ms = max(1, int(self.tool_pending_ttl_seconds * 1000))
        state["pending_tool_name"] = str(tool_name or "")
        state["pending_tool_deadline_ms"] = int(time.time() * 1000) + ttl_ms

    def _collect_recent_tool_slots(self, tool_name: str, messages: list, lookback: int = 8) -> dict:
        if not tool_name or not messages:
            return {}
        required_params = self.assessment_profiles.get(tool_name, {}).get("required_params", [])
        if not required_params:
            return {}
        collected = {}
        for message in reversed(messages[-lookback:]):
            content = self._extract_message_content(message)
            if not content:
                continue
            extracted = self._extract_required_slots(tool_name, content)
            if not extracted:
                continue
            for key, value in extracted.items():
                if key not in collected:
                    collected[key] = value
            if all(param in collected for param in required_params):
                break
        return collected

    def _extract_diabetes_slots(self, question: str) -> dict:
        slots = {}
        age_match = re.search(r"(?:年龄|age)\s*[:：]?\s*(\d{1,3})", question, re.IGNORECASE)
        if not age_match:
            age_match = re.search(r"(\d{1,3})\s*岁", question)
        if age_match:
            slots["age"] = int(age_match.group(1))
        bmi_match = re.search(r"(?:bmi|体重指数)\s*[:：]?\s*(\d{1,2}(?:\.\d{1,2})?)", question, re.IGNORECASE)
        if bmi_match:
            slots["bmi"] = float(bmi_match.group(1))
        return slots

    def _extract_required_slots(self, tool_name: str, question: str) -> dict:
        if tool_name == "hypertension_risk_assessment":
            return self._extract_hypertension_slots(question)
        if tool_name == "diabetes_risk_assessment":
            return self._extract_diabetes_slots(question)
        return {}

    def _missing_required_params(self, tool_name: str, question: str) -> list[str]:
        profile = self.assessment_profiles.get(tool_name, {})
        required_params = profile.get("required_params", [])
        if not required_params:
            return []
        extracted_slots = self._extract_required_slots(tool_name, question)
        return [param for param in required_params if param not in extracted_slots]

    def _is_assessment_skill_candidate(self, question: str) -> bool:
        question_text = (question or "").lower()
        generic_keywords = ["评估", "风险", "风险评估", "计算"]
        if any(keyword in question_text for keyword in generic_keywords):
            return True
        for profile in self.assessment_profiles.values():
            for keyword in profile.get("trigger_keywords", []):
                if keyword.lower() in question_text:
                    return True
        return False

    def _build_missing_params_answer(self, tool_name: str, missing_params: list[str]) -> str:
        display_name = self.assessment_profiles.get(tool_name, {}).get("display_name", tool_name)
        missing_text = "、".join(missing_params) if missing_params else "必需参数"
        return f"要进行{display_name}，还需要补充以下参数：{missing_text}。请补充后我将自动继续评估。"

    def _validate_tool_args(self, tool_name: str, tool_args: dict) -> dict:
        errors = {}
        if tool_name == "hypertension_risk_assessment":
            age = tool_args.get("age")
            systolic_bp = tool_args.get("systolic_bp")
            diastolic_bp = tool_args.get("diastolic_bp")
            if not isinstance(age, int) or age < 1 or age > 120:
                errors["age"] = "年龄需为1-120的整数"
            if not isinstance(systolic_bp, int) or systolic_bp < 60 or systolic_bp > 260:
                errors["systolic_bp"] = "收缩压需为60-260的整数"
            if not isinstance(diastolic_bp, int) or diastolic_bp < 30 or diastolic_bp > 180:
                errors["diastolic_bp"] = "舒张压需为30-180的整数"
        elif tool_name == "diabetes_risk_assessment":
            age = tool_args.get("age")
            bmi = tool_args.get("bmi")
            if not isinstance(age, int) or age < 1 or age > 120:
                errors["age"] = "年龄需为1-120的整数"
            if not isinstance(bmi, (int, float)) or bmi < 10 or bmi > 80:
                errors["bmi"] = "BMI需为10-80之间的数值"
        return errors

    def check_tool_needed_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: CHECK_TOOL_NEEDED - 技能选择与工具门控")
        state["need_tool"] = False
        state["selected_skill"] = ""
        state["selected_tool"] = ""
        state["tool_missing_params"] = []
        state["tool_prefilled_args"] = {}
        state["tool_selection_reason"] = ""
        state["tool_clarify_message"] = ""
        if not self.tools:
            return state

        messages = state.get("messages", [])
        if not messages:
            return state

        user_question = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        pending_tool = ""
        now_ms = int(time.time() * 1000)
        state_pending_tool = str(state.get("pending_tool_name") or "").strip()
        state_pending_deadline = self._safe_int(state.get("pending_tool_deadline_ms"), 0)
        if state_pending_tool and state_pending_deadline <= now_ms:
            self._clear_pending_tool_state(state)
            state_pending_tool = ""
        if state_pending_tool in self.tool_map and state_pending_deadline > now_ms:
            pending_tool = state_pending_tool
        if pending_tool and not self._is_pending_followup_for_tool(pending_tool, user_question):
            pending_tool = ""
        history_pending_tool = self._resolve_pending_assessment_tool(messages[:-1])
        if not pending_tool and history_pending_tool and self._is_pending_followup_for_tool(history_pending_tool, user_question):
            pending_tool = history_pending_tool
        if pending_tool and not self._is_pending_followup_for_tool(pending_tool, user_question):
            pending_tool = ""
        if not self._is_assessment_skill_candidate(user_question) and not pending_tool:
            self.logger.info("未命中评估技能轻量门控，跳过技能激活")
            self._clear_pending_tool_state(state)
            return state

        available_profile_lines = []
        for tool_name, profile in self.assessment_profiles.items():
            if tool_name in self.tool_map:
                available_profile_lines.append(
                    f"- {tool_name}({profile['display_name']}), 必需参数: {', '.join(profile['required_params'])}"
                )
        profiles_text = "\n".join(available_profile_lines)
        if not profiles_text:
            return state
        selected_tool = pending_tool
        missing_params = []
        selection_reason = "承接上一轮补参流程" if pending_tool else ""
        try:
            prompt = f"""你是健康风险评估技能路由器。请判断是否激活评估技能，并选择最匹配的工具。
可用评估工具:
{profiles_text}
用户问题: {user_question}
要求:
1) 只有当用户明确要风险评估/计算，或提供了可评估参数时，use_assessment_skill=true
2) selected_tool只能从可用评估工具里选择
3) missing_params只包含必需参数缺失项
4) 如果不激活技能，selected_tool返回null"""
            if not pending_tool:
                structured_llm = self.llm.with_structured_output(ToolSkillDecision)
                decision = structured_llm.invoke(prompt)
                selected_tool = (decision.selected_tool or "").strip()
                if selected_tool and selected_tool not in self.tool_map:
                    selected_tool = ""
                selection_reason = decision.reasoning
            if not selected_tool:
                if any(keyword in user_question for keyword in ["高血压", "血压", "收缩压", "舒张压", "高压", "低压"]):
                    selected_tool = "hypertension_risk_assessment" if "hypertension_risk_assessment" in self.tool_map else ""
                elif any(keyword in user_question.lower() for keyword in ["糖尿病", "血糖", "bmi", "体重指数", "diabetes", "glucose"]):
                    selected_tool = "diabetes_risk_assessment" if "diabetes_risk_assessment" in self.tool_map else ""
            if selected_tool:
                prefilled_args = self._collect_recent_tool_slots(selected_tool, messages)
                required_params = self.assessment_profiles.get(selected_tool, {}).get("required_params", [])
                missing_params = [param for param in required_params if param not in prefilled_args]
                state["tool_prefilled_args"] = prefilled_args
            state["selected_skill"] = self.assessment_skill_name if selected_tool else ""
            state["selected_tool"] = selected_tool
            state["tool_missing_params"] = missing_params
            state["tool_selection_reason"] = selection_reason
            state["need_tool"] = bool(state["selected_skill"] and state["selected_tool"])
            if not state["selected_tool"]:
                self._clear_pending_tool_state(state)
                state["selected_skill"] = self.assessment_skill_name
                state["tool_selection_reason"] = "需要明确评估类型"
                state["tool_clarify_message"] = "请说明要评估的类型（高血压风险评估 / 糖尿病风险评估），并提供对应参数，例如年龄、血压或BMI。"
                state["need_tool"] = True
            elif missing_params:
                self._set_pending_tool_state(state, state["selected_tool"])
            else:
                self._clear_pending_tool_state(state)
            self.logger.info(
                f"技能路由: need_tool={state['need_tool']}, skill={state['selected_skill']}, tool={state['selected_tool']}, missing={missing_params}"
            )
        except Exception as e:
            self.logger.error(f"技能路由失败: {e}")
            state["need_tool"] = False
            state["tool_prefilled_args"] = {}
            self._clear_pending_tool_state(state)
        return state

    def tool_calling_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        self.logger.info("="*50)
        self.logger.info("[RAG Graph] 节点: TOOL_CALLING - 执行工具调用")
        messages = state.get("messages", [])
        user_question = messages[-1].content if messages and hasattr(messages[-1], "content") else ""
        selected_skill = state.get("selected_skill", "")
        selected_tool = state.get("selected_tool")
        prefilled_args = state.get("tool_prefilled_args", {}) or {}
        self.logger.info(f"用户问题: {user_question}")
        self.logger.info(f"选中的技能: {selected_skill}")
        self.logger.info(f"选中的工具: {selected_tool}")
        self.logger.info(f"预提取参数: {prefilled_args}")
        clarify_message = state.get("tool_clarify_message")
        if clarify_message:
            self._clear_pending_tool_state(state)
            safe_answer = self._apply_medical_safety_notice(user_question, clarify_message, force_medical=True)
            state["final_answer"] = safe_answer
            state["messages"] = [AIMessage(content=safe_answer)]
            return state
        missing_params = state.get("tool_missing_params", [])
        if missing_params:
            self._set_pending_tool_state(state, selected_tool or "")
            answer = self._build_missing_params_answer(selected_tool or "", missing_params)
            safe_answer = self._apply_medical_safety_notice(user_question, answer, force_medical=True)
            state["final_answer"] = safe_answer
            state["messages"] = [AIMessage(content=safe_answer)]
            return state
        if not selected_tool or selected_tool not in self.tool_map:
            available_tools = list(self.tool_map.keys())
            error = ToolNotFoundError(tool_name=selected_tool or "unknown", available_tools=available_tools)
            state["error"] = error.to_dict()
            safe_answer = self._apply_medical_safety_notice(user_question, error.message, force_medical=True)
            state["final_answer"] = safe_answer
            state["messages"] = [AIMessage(content=safe_answer)]
            return state
        bound_tool = self.tool_map[selected_tool]
        required_params = self.assessment_profiles.get(selected_tool, {}).get("required_params", [])
        required_params_text = ", ".join(required_params) if required_params else "无"
        try:
            llm_with_tools = self.llm.bind_tools([bound_tool])
            prompt = f"""你是健康评估技能执行器。
用户问题：{user_question}
已选择技能：{selected_skill}
你必须且只能调用工具：{selected_tool}
必需参数：{required_params_text}
历史上下文已提取参数：{json.dumps(prefilled_args, ensure_ascii=False)}
请从用户问题中抽取参数并调用工具。"""
            self.logger.info("调用LLM，期待工具调用...")
            response = llm_with_tools.invoke(prompt)
            if hasattr(response, 'tool_calls') and response.tool_calls:
                self.logger.info(f"LLM请求调用 {len(response.tool_calls)} 个工具")
                audit_logger = get_audit_logger()
                target_calls = [call for call in response.tool_calls if call.get("name") == selected_tool]
                if not target_calls:
                    safe_answer = self._apply_medical_safety_notice(
                        user_question,
                        f"评估执行失败：模型未按要求调用 {selected_tool}。请重试或补充更明确参数。",
                        force_medical=True
                    )
                    state["final_answer"] = safe_answer
                    state["messages"] = [AIMessage(content=safe_answer)]
                    return state
                selected_call = target_calls[0]
                model_args = selected_call.get("args", {}) or {}
                if not isinstance(model_args, dict):
                    model_args = {}
                tool_args = {**prefilled_args, **model_args}
                validation_errors = self._validate_tool_args(selected_tool, tool_args)
                if validation_errors:
                    self._set_pending_tool_state(state, selected_tool)
                    error = ToolValidationError(
                        tool_name=selected_tool,
                        validation_errors=validation_errors,
                        details="工具参数预校验失败"
                    )
                    state["error"] = error.to_dict()
                    safe_answer = self._apply_medical_safety_notice(
                        user_question,
                        f"{error.message}，请补充或修正参数后重试。",
                        force_medical=True
                    )
                    state["final_answer"] = safe_answer
                    state["messages"] = [AIMessage(content=safe_answer)]
                    return state
                start_time = time.time()
                tool_result = None
                tool_error = None
                try:
                    tool_result = self._execute_tool_with_timeout(bound_tool, tool_args)
                    self.logger.info(f"工具执行成功: {selected_tool}")
                except ToolExecutionTimeoutError as timeout_error:
                    tool_error = timeout_error.to_dict()
                    state["error"] = tool_error
                    safe_answer = self._apply_medical_safety_notice(user_question, timeout_error.message, force_medical=True)
                    state["final_answer"] = safe_answer
                    state["messages"] = [AIMessage(content=safe_answer)]
                    return state
                except Exception as tool_exec_error:
                    error = ToolExecutionError(
                        tool_name=selected_tool,
                        error_message=str(tool_exec_error),
                        details=f"{type(tool_exec_error).__name__}: {tool_exec_error}"
                    )
                    tool_error = error.to_dict()
                    state["error"] = tool_error
                    safe_answer = self._apply_medical_safety_notice(user_question, error.message, force_medical=True)
                    state["final_answer"] = safe_answer
                    state["messages"] = [AIMessage(content=safe_answer)]
                    return state
                finally:
                    execution_time_ms = (time.time() - start_time) * 1000
                    audit_logger.log_tool_call(
                        conversation_id=state.get("session_id", "unknown"),
                        user_id=state.get("user_id", "unknown"),
                        tool_name=selected_tool,
                        tool_args=tool_args,
                        result=tool_result,
                        error=tool_error,
                        execution_time_ms=execution_time_ms,
                        metadata={"selected_skill": selected_skill, "selection_reason": state.get("tool_selection_reason", "")}
                    )
                final_prompt = f"""基于以下工具执行结果，请给用户一个清晰、专业、友好的回答。
用户问题：{user_question}
工具结果：
{json.dumps(tool_result, ensure_ascii=False, indent=2)}
请用通俗语言解释结论并给出可执行建议。"""
                final_response = self.llm.invoke(final_prompt)
                safe_answer = self._apply_medical_safety_notice(user_question, final_response.content, force_medical=True)
                state["final_answer"] = safe_answer
                self._clear_pending_tool_state(state)
                state["answer_sources"] = [{
                    "index": 1,
                    "tool_name": selected_tool,
                    "document_name": f"工具: {selected_tool}",
                    "content": f"风险等级: {tool_result.get('risk_level', 'N/A')}, 评分: {tool_result.get('risk_score', 'N/A')}",
                    "retrieval_mode": "tool",
                    "metadata": {
                        "tool_args": tool_args,
                        "risk_level": tool_result.get("risk_level"),
                        "risk_score": tool_result.get("risk_score"),
                        "risk_factors": tool_result.get("risk_factors", []),
                        "recommendations": tool_result.get("recommendations", []),
                        "selected_skill": selected_skill
                    }
                }]
                state["messages"] = [AIMessage(content=safe_answer)]
                return state
            safe_answer = self._apply_medical_safety_notice(
                user_question,
                "抱歉，我无法从您的问题中提取足够的参数来进行评估。请提供更详细的信息。",
                force_medical=True
            )
            state["final_answer"] = safe_answer
            state["messages"] = [AIMessage(content=safe_answer)]
        except ToolCallException as e:
            self.logger.error(f"工具调用异常: {e.to_dict()}")
            state["error"] = e.to_dict()
            safe_answer = self._apply_medical_safety_notice(user_question, e.message, force_medical=True)
            state["final_answer"] = safe_answer
            state["messages"] = [AIMessage(content=safe_answer)]
        except Exception as e:
            import traceback
            self.logger.error(f"工具调用失败: {e}")
            self.logger.error(traceback.format_exc())
            from backend.agent.tools.exceptions import ToolCallErrorCode
            error = ToolCallException(
                code=ToolCallErrorCode.UNKNOWN_ERROR,
                message="抱歉，工具调用时遇到未知问题",
                details=traceback.format_exc(),
                metadata={"original_error": str(e)}
            )
            state["error"] = error.to_dict()
            safe_answer = self._apply_medical_safety_notice(user_question, error.message, force_medical=True)
            state["final_answer"] = safe_answer
            state["messages"] = [AIMessage(content=safe_answer)]
        return state

    def check_retrieval_needed_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        """判断是否需要检索节点

        根据retrieval_mode和LLM判断是否需要进行检索：
        1. 如果retrieval_mode为NO_RETRIEVAL，直接设置为False
        2. 否则调用LLM进行智能判断

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态
        """
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: CHECK_RETRIEVAL_NEEDED - 判断是否需要检索")
        
        retrieval_mode = state['retrieval_mode']
        latest_message = self._extract_latest_message(state)
        decision_stats = {
            "stage": "",
            "reason": "",
            "rule_result": None,
            "classifier_result": None,
            "llm_result": None,
            "rule_library_version": self.rule_library_version,
            "rule_library_source": "",
            "rule_conflict_count": 0,
            "decision_path": []
        }


        # 检查retrieval_mode是否为NO_RETRIEVAL
        if retrieval_mode == RetrievalMode.NO_RETRIEVAL:
            self.logger.info("跳过检索")
            state["need_retrieval"] = False
            state["need_retrieval_reason"] = "用户设置为不需要检索模式"
            state["original_question"] = latest_message
            decision_stats["stage"] = "mode_no_retrieval"
            decision_stats["reason"] = state["need_retrieval_reason"]
            state["retrieval_decision_stats"] = decision_stats
            return state

        if retrieval_mode != RetrievalMode.AUTO:
            self.logger.info("直接设置需要检索")
            state["need_retrieval"] = True
            state["need_retrieval_reason"] = f"用户设置为{retrieval_mode}检索模式，直接进行检索"
            state["original_question"] = latest_message
            decision_stats["stage"] = "mode_force_retrieval"
            decision_stats["reason"] = state["need_retrieval_reason"]
            state["retrieval_decision_stats"] = decision_stats
            return state

        self.logger.info("AUTO模式，规则优先+轻量分类器+LLM判定...")
        try:
            rule_library = self._get_rule_library()
            rule_library_meta = rule_library.get("meta", {})
            decision_stats["rule_library_version"] = rule_library_meta.get("version", self.rule_library_version)
            decision_stats["rule_library_source"] = rule_library_meta.get("source", "")
            decision_stats["rule_conflict_count"] = self._safe_int(rule_library_meta.get("conflict_count", 0), 0)
            rule_result = self._rule_first_retrieval_decision(latest_message, library=rule_library)
            decision_stats["rule_result"] = rule_result
            decision_stats["decision_path"] = list(rule_result.get("decision_path", []))
            if rule_result.get("decision") is not None:
                state["need_retrieval"] = bool(rule_result.get("decision"))
                state["need_retrieval_reason"] = rule_result.get("reason", "")
                state["original_question"] = latest_message
                decision_stats["stage"] = "rule"
                decision_stats["reason"] = state["need_retrieval_reason"]
                state["retrieval_decision_stats"] = decision_stats
                return state

            classifier_result = self._lightweight_retrieval_classifier(latest_message)
            decision_stats["classifier_result"] = classifier_result
            if classifier_result.get("decision") is not None:
                state["need_retrieval"] = bool(classifier_result.get("decision"))
                state["need_retrieval_reason"] = classifier_result.get("reason", "")
                state["original_question"] = latest_message
                decision_stats["stage"] = "lightweight_classifier"
                decision_stats["reason"] = state["need_retrieval_reason"]
                decision_stats["decision_path"] = decision_stats["decision_path"] + ["fallback:lightweight_classifier"]
                state["retrieval_decision_stats"] = decision_stats
                return state

            prompt_template = RAGGraphPrompts.get_retrieval_need_judgment_prompt()
            prompt = prompt_template.format(question=latest_message)
            structured_llm = self.llm.with_structured_output(RetrievalNeedDecision)
            decision = structured_llm.invoke(prompt)
            state["need_retrieval"] = decision.need_retrieval
            state["need_retrieval_reason"] = decision.reasoning
            if decision.extracted_question and decision.extracted_question.strip():
                state["original_question"] = decision.extracted_question.strip()
            else:
                state["original_question"] = latest_message
            decision_stats["llm_result"] = {
                "decision": bool(decision.need_retrieval),
                "reason": decision.reasoning
            }
            decision_stats["stage"] = "llm"
            decision_stats["reason"] = decision.reasoning
            decision_stats["decision_path"] = decision_stats["decision_path"] + ["fallback:llm"]
            state["retrieval_decision_stats"] = decision_stats
        except Exception as e:
            self.logger.error(f"检索需求判断失败: {e}")
            state["need_retrieval"] = True
            state["need_retrieval_reason"] = f"检索需求判断过程出错: {str(e)}"
            state["original_question"] = latest_message
            decision_stats["stage"] = "fallback_error"
            decision_stats["reason"] = state["need_retrieval_reason"]
            decision_stats["decision_path"] = decision_stats["decision_path"] + ["fallback:error"]
            state["retrieval_decision_stats"] = decision_stats

        return state

    def expand_subquestions_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        """由原始问题扩展子问题节点

        根据原始问题，调用LLM将其分解为多个具体的子问题，
        以便进行更精确的信息检索。

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态，包含扩展的子问题列表
        """
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: EXPAND_SUBQUESTIONS - 扩展子问题")

        # 获取原始问题
        original_question = state.get("original_question", "")
        rule_library = self._get_rule_library()
        subquestion_policy = rule_library.get("subquestion_policy", {})
        rule_library_meta = rule_library.get("meta", {})
        configured_max_count = self._policy_int("subquestion_max_count", self._safe_int(subquestion_policy.get("max_count", self.subquestion_max_count), self.subquestion_max_count))
        configured_trigger_min_chars = self._policy_int(
            "subquestion_trigger_min_chars",
            self._safe_int(subquestion_policy.get("trigger_min_chars", self.subquestion_trigger_min_chars), self.subquestion_trigger_min_chars)
        )
        expansion_stats = {
            "enabled": self._policy_bool("enable_subquestion_expansion", self.enable_subquestion_expansion),
            "triggered": False,
            "reason": "",
            "max_count": configured_max_count,
            "trigger_min_chars": configured_trigger_min_chars,
            "policy_id": str(subquestion_policy.get("id", "subquestion.default")),
            "policy_version": str(subquestion_policy.get("version", self.rule_library_version)),
            "policy_scope": str(subquestion_policy.get("scope", self.retrieval_cache_scope)),
            "policy_source": str(rule_library_meta.get("source", "")),
            "count_before_limit": 0,
            "count_after_limit": 0,
            "complexity": {}
        }
        if not original_question:
            self.logger.warning("原始问题为空，无法扩展子问题")
            state["subquestions"] = []
            expansion_stats["reason"] = "原始问题为空"
            state["subquestion_expansion_stats"] = expansion_stats
            return state

        if not expansion_stats["enabled"]:
            state["subquestions"] = []
            expansion_stats["reason"] = "子问题扩展开关关闭"
            state["subquestion_expansion_stats"] = expansion_stats
            return state

        normalized_question = original_question.strip()
        complexity_result = self._compute_subquestion_complexity(normalized_question)
        expansion_stats["complexity"] = complexity_result
        trigger_by_length = len(normalized_question) >= configured_trigger_min_chars
        if not (complexity_result.get("triggered") or trigger_by_length):
            state["subquestions"] = []
            expansion_stats["reason"] = "未达到复杂度触发门槛"
            state["subquestion_expansion_stats"] = expansion_stats
            return state

        self.logger.info(f"原始问题: {original_question}")
        expansion_stats["triggered"] = True
        expansion_stats["reason"] = "命中复杂度或长度触发门槛"

        try:
            # 获取子问题扩展提示词
            prompt_template = RAGGraphPrompts.get_subquestion_expansion_prompt()
            prompt = prompt_template.format(question=original_question)

            # 使用结构化输出调用LLM
            structured_llm = self.llm.with_structured_output(SubquestionExpansion)
            expansion_result = structured_llm.invoke(prompt)

            # 获取子问题列表
            subquestions = expansion_result.subquestions

            # 验证和清理子问题
            if subquestions:
                # 过滤空字符串和重复项
                cleaned_subquestions = []
                for sq in subquestions:
                    if sq and sq.strip() and sq.strip() not in cleaned_subquestions:
                        cleaned_subquestions.append(sq.strip())
                expansion_stats["count_before_limit"] = len(cleaned_subquestions)
                limited_subquestions = cleaned_subquestions[:max(expansion_stats["max_count"], 0)]
                expansion_stats["count_after_limit"] = len(limited_subquestions)
                state["subquestions"] = limited_subquestions
                self.logger.info(f"成功扩展 {len(limited_subquestions)} 个子问题")
            else:
                state["subquestions"] = []
                self.logger.info("未生成子问题，跳过扩展")

        except Exception as e:
            self.logger.error(f"子问题扩展失败: {e}")
            state["subquestions"] = []
            expansion_stats["reason"] = f"扩展失败: {str(e)}"

        state["subquestion_expansion_stats"] = expansion_stats

        return state

    def classify_question_type_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        """判断检索类型节点

        根据context中的retrieval_mode配置决定使用哪种检索方式。
        如果是AUTO模式，调用LLM进行智能判断并更新retrieval_mode。

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态
        """
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: CLASSIFY_QUESTION_TYPE - 判断检索类型")


        # 如果是AUTO模式，调用LLM进行智能判断
        if state["retrieval_mode"] == RetrievalMode.AUTO:
            self.logger.info("AUTO模式，使用向量优先+按需图检索策略")
            state["retrieval_mode"] = RetrievalMode.HYBRID
            state["retrieval_mode_reason"] = "AUTO模式统一走融合检索，向量优先并按条件补充图检索"
        else:
            self.logger.info(f"未进行智能判断，保持当前检索模式: {state['retrieval_mode']}")
            state["retrieval_mode_reason"] = f"未进行智能判断,保持当前检索模式{state['retrieval_mode']}"

        self.logger.info(f"最终检索模式: {state['retrieval_mode']}")
        return state

    def _deduplicate_retrieved_docs(self, docs: list[RetrievedDocument]) -> list[RetrievedDocument]:
        unique_docs = []
        seen_keys = set()
        for doc in docs:
            metadata = doc.metadata or {}
            pk = metadata.get("pk") or metadata.get("id")
            if pk is not None:
                key = ("pk", str(pk))
            else:
                document_name = metadata.get("document_name", "")
                chunk_index = metadata.get("chunk_index", "")
                key = ("fallback", str(document_name), str(chunk_index), doc.page_content[:80])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_docs.append(doc)
        return unique_docs

    def _is_near_duplicate(self, left: str, right: str) -> bool:
        if not left or not right:
            return False
        normalized_left = "".join(ch.lower() for ch in left if not ch.isspace())
        normalized_right = "".join(ch.lower() for ch in right if not ch.isspace())
        if not normalized_left or not normalized_right:
            return False
        if normalized_left[:180] == normalized_right[:180]:
            return True
        left_set = set(normalized_left[:800])
        right_set = set(normalized_right[:800])
        if not left_set or not right_set:
            return False
        overlap = len(left_set & right_set)
        union = len(left_set | right_set)
        if union == 0:
            return False
        return (overlap / union) >= 0.92

    def _deduplicate_semantic_docs(self, docs: list[RetrievedDocument]) -> list[RetrievedDocument]:
        deduped_docs: list[RetrievedDocument] = []
        for doc in docs:
            is_dup = False
            current_name = (doc.metadata or {}).get("document_name", "")
            for existing_doc in deduped_docs:
                existing_name = (existing_doc.metadata or {}).get("document_name", "")
                if current_name and existing_name and current_name != existing_name:
                    continue
                if self._is_near_duplicate(doc.page_content, existing_doc.page_content):
                    is_dup = True
                    break
            if not is_dup:
                deduped_docs.append(doc)
        return deduped_docs

    def _extract_overlap_score(self, query_text: str, content: str) -> float:
        if not query_text or not content:
            return 0.0
        query_tokens = set(ch.lower() for ch in query_text if not ch.isspace())
        content_tokens = set(ch.lower() for ch in content[:1000] if not ch.isspace())
        if not query_tokens:
            return 0.0
        overlap = len(query_tokens & content_tokens)
        return overlap / len(query_tokens)

    def _estimate_vector_confidence(self, query_text: str, docs: list[RetrievedDocument]) -> float:
        if not query_text or not docs:
            return 0.0
        top_docs = docs[: min(3, len(docs))]
        weighted_score = 0.0
        total_weight = 0.0
        for idx, doc in enumerate(top_docs):
            weight = 1.0 / (idx + 1)
            metadata = doc.metadata or {}
            overlap_score = self._extract_overlap_score(query_text, doc.page_content)
            semantic_score = max(0.0, float(metadata.get("semantic_score", 0.0)))
            rrf_score = max(0.0, float(metadata.get("rrf_score", 0.0)))
            fused = 0.75 * max(overlap_score, semantic_score) + 0.25 * min(rrf_score, 1.0)
            weighted_score += weight * fused
            total_weight += weight
        if total_weight == 0.0:
            return 0.0
        return round(weighted_score / total_weight, 6)

    def _get_chunk_role_weight(self, metadata: dict | None) -> float:
        chunk_role = str((metadata or {}).get("chunk_role", "general_medical"))
        return float(self.chunk_role_weights.get(chunk_role, self.chunk_role_weights.get("general_medical", 1.0)))

    def _get_medical_priority_bonus(self, metadata: dict | None) -> float:
        safe_metadata = metadata or {}
        bonus = 0.0
        if bool(safe_metadata.get("is_key_clause", False)):
            bonus += self.key_clause_bonus
        evidence_level = str(safe_metadata.get("evidence_level") or "").upper()
        if evidence_level:
            bonus += self.evidence_level_bonus
            if evidence_level in {"I", "IA", "IB", "A", "A+", "A-"}:
                bonus += self.evidence_level_bonus
        return bonus

    def _content_similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        left_tokens = set(ch.lower() for ch in left[:1000] if not ch.isspace())
        right_tokens = set(ch.lower() for ch in right[:1000] if not ch.isspace())
        if not left_tokens or not right_tokens:
            return 0.0
        inter = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        if union == 0:
            return 0.0
        return inter / union

    def _mmr_select_docs(self, docs: list[RetrievedDocument], select_k: int) -> list[RetrievedDocument]:
        if select_k <= 0 or not docs:
            return []
        candidate_docs = list(docs)
        selected_docs: list[RetrievedDocument] = []
        while candidate_docs and len(selected_docs) < select_k:
            best_doc = None
            best_score = None
            for doc in candidate_docs:
                relevance = float((doc.metadata or {}).get("rrf_score", 0.0))
                if not selected_docs:
                    mmr_score = relevance
                else:
                    redundancy = max(
                        self._content_similarity(doc.page_content, selected.page_content)
                        for selected in selected_docs
                    )
                    mmr_score = self.mmr_lambda * relevance - (1.0 - self.mmr_lambda) * redundancy
                if best_score is None or mmr_score > best_score:
                    best_score = mmr_score
                    best_doc = doc
            if best_doc is None:
                break
            selected_docs.append(best_doc)
            candidate_docs.remove(best_doc)
        return selected_docs

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot_value = sum(l * r for l, r in zip(left, right))
        left_norm = math.sqrt(sum(l * l for l in left))
        right_norm = math.sqrt(sum(r * r for r in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot_value / (left_norm * right_norm)

    def _semantic_rerank_docs(self, query_text: str, docs: list[RetrievedDocument]) -> tuple[list[RetrievedDocument], dict]:
        rerank_stats = {
            "enabled": self.enable_semantic_rerank,
            "attempted": False,
            "fallback": False,
            "input_total": len(docs or []),
            "input_used": 0,
            "input_limit": self.semantic_rerank_max_inputs
        }
        if not self.enable_semantic_rerank or not self.embedding_model or not query_text or not docs:
            return docs, rerank_stats
        limited_size = max(self._policy_int("semantic_rerank_max_inputs", self.semantic_rerank_max_inputs), 1)
        rerank_candidates = list(docs[:limited_size])
        tail_candidates = list(docs[limited_size:])
        rerank_stats["attempted"] = True
        rerank_stats["input_used"] = len(rerank_candidates)
        self.semantic_rerank_total_count += 1
        try:
            query_vector = self.embedding_model.embed_query(query_text)
            doc_texts = [doc.page_content for doc in rerank_candidates]
            doc_vectors = self.embedding_model.embed_documents(doc_texts)
            scored_docs = []
            for doc, vector in zip(rerank_candidates, doc_vectors):
                semantic_score = self._cosine_similarity(query_vector, vector)
                metadata = dict(doc.metadata or {})
                metadata["semantic_score"] = round(semantic_score, 6)
                metadata["medical_priority"] = round(
                    self._get_chunk_role_weight(metadata) + self._get_medical_priority_bonus(metadata),
                    6
                )
                scored_docs.append(
                    RetrievedDocument(
                        page_content=doc.page_content,
                        metadata=metadata
                    )
                )
            scored_docs.sort(
                key=lambda d: (
                    float((d.metadata or {}).get("semantic_score", 0.0)),
                    float((d.metadata or {}).get("medical_priority", 1.0)),
                    float((d.metadata or {}).get("rrf_score", 0.0))
                ),
                reverse=True
            )
            return scored_docs + tail_candidates, rerank_stats
        except Exception as exc:
            self.semantic_rerank_fallback_count += 1
            rerank_stats["fallback"] = True
            self.logger.warning(f"语义重排失败，回退至RRF排序: {exc}")
            return docs, rerank_stats

    def _split_graph_result_to_docs(self, content: str, max_docs: int) -> list[RetrievedDocument]:
        stripped = (content or "").strip()
        if not stripped:
            return []
        raw_segments = [seg.strip() for seg in stripped.split("\n\n") if seg.strip()]
        if not raw_segments:
            raw_segments = [stripped]
        docs: list[RetrievedDocument] = []
        limit = max(max_docs * 2, 4)
        for idx, segment in enumerate(raw_segments[:limit]):
            docs.append(
                RetrievedDocument(
                    page_content=segment,
                    metadata={
                        "source": "lightrag_graph",
                        "retrieval_mode": "global",
                        "document_name": "知识图谱检索结果",
                        "chunk_index": idx,
                        "rank_graph": idx + 1
                    }
                )
            )
        return docs

    def _merge_retrieved_docs(
        self,
        vector_docs: list[RetrievedDocument],
        graph_docs: list[RetrievedDocument],
        max_docs: int,
        query_text: str
    ) -> tuple[list[RetrievedDocument], dict]:
        rrf_k = self.rrf_k
        source_weights = {
            "vector": self.vector_source_weight,
            "graph": self.graph_source_weight
        }
        score_map: dict[tuple, float] = {}
        doc_map: dict[tuple, RetrievedDocument] = {}
        source_map: dict[tuple, set[str]] = {}

        def doc_key(doc: RetrievedDocument) -> tuple:
            metadata = doc.metadata or {}
            pk = metadata.get("pk") or metadata.get("id")
            if pk is not None:
                return ("pk", str(pk))
            return (
                "fallback",
                str(metadata.get("document_name", "")),
                str(metadata.get("chunk_index", "")),
                doc.page_content[:120]
            )

        def add_with_rrf(docs: list[RetrievedDocument], source_tag: str) -> None:
            for rank_index, doc in enumerate(docs or []):
                key = doc_key(doc)
                score = source_weights[source_tag] * (1.0 / (rrf_k + rank_index + 1))
                lexical_bonus = 0.08 * self._extract_overlap_score(query_text, doc.page_content)
                metadata = dict(doc.metadata or {})
                role_weight = self._get_chunk_role_weight(metadata)
                medical_bonus = self._get_medical_priority_bonus(metadata)
                total = (score + lexical_bonus) * role_weight + medical_bonus
                score_map[key] = score_map.get(key, 0.0) + total
                if key not in doc_map:
                    doc_map[key] = doc
                source_set = source_map.get(key, set())
                source_set.add(source_tag)
                source_map[key] = source_set

        add_with_rrf(vector_docs, "vector")
        add_with_rrf(graph_docs, "graph")

        ranked_items = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
        ranked_docs: list[RetrievedDocument] = []
        for rank_idx, (key, score) in enumerate(ranked_items):
            doc = doc_map[key]
            metadata = dict(doc.metadata or {})
            metadata["rrf_score"] = round(score, 6)
            metadata["fused_rank"] = rank_idx + 1
            metadata["source"] = "hybrid_fused"
            metadata["fused_sources"] = sorted(list(source_map.get(key, set())))
            ranked_docs.append(RetrievedDocument(page_content=doc.page_content, metadata=metadata))

        ranked_docs = self._deduplicate_retrieved_docs(ranked_docs)
        ranked_docs = self._deduplicate_semantic_docs(ranked_docs)
        ranked_docs, rerank_stats = self._semantic_rerank_docs(query_text, ranked_docs)

        text_docs = [doc for doc in ranked_docs if (doc.metadata or {}).get("chunk_type") != "chart"]
        chart_docs = [doc for doc in ranked_docs if (doc.metadata or {}).get("chunk_type") == "chart"]

        vector_text_docs = []
        graph_text_docs = []
        neutral_text_docs = []
        for doc in text_docs:
            fused_sources = set((doc.metadata or {}).get("fused_sources") or [])
            if "graph" in fused_sources and "vector" in fused_sources:
                neutral_text_docs.append(doc)
            elif "graph" in fused_sources:
                graph_text_docs.append(doc)
            else:
                vector_text_docs.append(doc)

        candidate_text_docs: list[RetrievedDocument] = []
        while vector_text_docs or graph_text_docs:
            if vector_text_docs:
                candidate_text_docs.append(vector_text_docs.pop(0))
            if graph_text_docs:
                candidate_text_docs.append(graph_text_docs.pop(0))
        candidate_text_docs.extend(neutral_text_docs)
        selected_text_docs = self._mmr_select_docs(candidate_text_docs, max_docs)

        max_chart_docs = 2 if max_docs >= 2 else 1
        selected_chart_docs = chart_docs[:max_chart_docs]

        fused_docs = selected_text_docs + selected_chart_docs
        if not fused_docs:
            fused_docs = ranked_docs[:max_docs]
        return fused_docs, rerank_stats

    def vector_db_retrieval_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        """向量数据库检索节点

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态
        """
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: VECTOR_DB_RETRIEVAL - 向量数据库检索")

        # 检查MilvusStorage是否可用
        if not self.milvus_storage:
            self.logger.warning("MilvusStorage未初始化，跳过向量检索")
            state["retrieved_docs"] = []
            state["vector_db_results"] = []
            return state

        # 获取检索查询
        original_question = state.get("original_question", "")
        if not original_question:
            self.logger.warning("未找到原始问题，跳过向量检索")
            state["retrieved_docs"] = []
            state["vector_db_results"] = []
            return state

        started_at = time.perf_counter()
        cache_hit = False
        try:
            # 从context获取检索配置
            context = runtime.context
            max_docs = context.max_retrieval_docs if context else 3
            subquestions = state.get("subquestions", [])
            cache_key = self._build_retrieval_cache_key(
                prefix="vector",
                query_text=original_question,
                subquestions=[],
                max_docs=max_docs
            )
            cached_payload = self._get_cached_item(self._vector_retrieval_cache, cache_key)
            if cached_payload is None:
                redis_payload = self._get_redis_retrieval_cache_sync(cache_key)
                if redis_payload is not None:
                    cached_payload = self._deserialize_vector_cache_payload(redis_payload)
                    self._set_cached_item(self._vector_retrieval_cache, cache_key, cached_payload)
            if cached_payload:
                cache_hit = True
                cached_retrieved_docs, cached_vector_results = cached_payload
                state["retrieved_docs"] = self._clone_retrieved_docs(cached_retrieved_docs)
                state["vector_db_results"] = self._clone_retrieved_docs(cached_vector_results)
                vector_confidence = self._estimate_vector_confidence(original_question, state["retrieved_docs"])
                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
                state["vector_retrieval_stats"] = {
                    "cache_hit": True,
                    "duration_ms": duration_ms,
                    "question_count": 1 + len(subquestions),
                    "retrieved_count": len(state["retrieved_docs"]),
                    "vector_candidate_count": len(state["vector_db_results"]),
                    "vector_confidence": vector_confidence
                }
                state["vector_confidence"] = vector_confidence
                self.logger.info("向量检索命中本地缓存")
                return state

            # 创建混合检索器
            hybrid_retriever = self.milvus_storage.create_hybrid_retriever(
                search_kwargs={"k": max(max_docs * 2, 6)}
            )

            # 收集所有需要检索的问题
            questions_to_search = [original_question]

            # 添加子问题到检索列表
            if subquestions:
                questions_to_search.extend(subquestions)
                self.logger.info(f"将对 {len(questions_to_search)} 个问题进行检索（1个原始问题 + {len(subquestions)}个子问题）")
            else:
                self.logger.info("将对原始问题进行检索")

            # 依次对每个问题进行检索
            all_retrieved_docs = []
            for i, question in enumerate(questions_to_search):
                try:
                    docs = hybrid_retriever.invoke(question)
                    all_retrieved_docs.extend(docs)
                except Exception as search_error:
                    self.logger.error(f"问题 {i+1} 检索失败: {search_error}")

            self.logger.info(f"总共检索到 {len(all_retrieved_docs)} 个文档")

            # 转换为RetrievedDocument格式
            converted_docs = []
            for doc in all_retrieved_docs:
                metadata = dict(doc.metadata or {})
                metadata["source"] = metadata.get("source") or "vector"
                retrieved_doc = RetrievedDocument(
                    page_content=doc.page_content,
                    metadata=metadata
                )
                converted_docs.append(retrieved_doc)
                # 调试日志：查看每个文档的metadata
                self.logger.info(f"检索文档 metadata: document_name={doc.metadata.get('document_name')}, "
                               f"chunk_index={doc.metadata.get('chunk_index')}, "
                               f"content_length={len(doc.page_content)}, "
                               f"pk={doc.metadata.get('pk')}")

            unique_docs = self._deduplicate_retrieved_docs(converted_docs)

            self.logger.info(f"去重后文档数: {len(unique_docs)}")

            text_docs = [doc for doc in unique_docs if doc.metadata.get("chunk_type") != "chart"]
            chart_docs = [doc for doc in unique_docs if doc.metadata.get("chunk_type") == "chart"]

            max_chart_docs = 2 if max_docs >= 2 else 1
            selected_text_docs = text_docs[:max_docs]
            selected_chart_docs = chart_docs[:max_chart_docs]
            fused_docs = selected_text_docs + selected_chart_docs
            if not fused_docs:
                fused_docs = unique_docs[:max_docs]

            self.logger.info(
                f"融合检索结果: 文本 {len(selected_text_docs)} 条, 图表 {len(selected_chart_docs)} 条, 总计 {len(fused_docs)} 条"
            )

            state["retrieved_docs"] = fused_docs
            state["vector_db_results"] = unique_docs
            vector_confidence = self._estimate_vector_confidence(original_question, fused_docs)
            state["vector_confidence"] = vector_confidence
            self._set_cached_item(
                self._vector_retrieval_cache,
                cache_key,
                (self._clone_retrieved_docs(fused_docs), self._clone_retrieved_docs(unique_docs))
            )
            self._set_redis_retrieval_cache_sync(
                cache_key,
                self._serialize_vector_cache_payload(
                    (self._clone_retrieved_docs(fused_docs), self._clone_retrieved_docs(unique_docs))
                )
            )
            cache_hit = False

        except Exception as e:
            self.logger.error(f"向量检索失败: {e}")
            # 检索失败时设置空结果
            state["retrieved_docs"] = []
            state["vector_db_results"] = []
            state["vector_confidence"] = 0.0
        finally:
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            state["vector_retrieval_stats"] = {
                "cache_hit": cache_hit,
                "duration_ms": duration_ms,
                "question_count": 1 + len(state.get("subquestions", []) or []),
                "retrieved_count": len(state.get("retrieved_docs") or []),
                "vector_candidate_count": len(state.get("vector_db_results") or []),
                "vector_confidence": float(state.get("vector_confidence") or 0.0)
            }

        return state

    async def hybrid_retrieval_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: HYBRID_RETRIEVAL - 融合检索")

        context = runtime.context
        max_docs = context.max_retrieval_docs if context else 3
        graph_budget_docs = self._resolve_graph_budget_docs(max_docs)
        graph_circuit_open = self._is_graph_circuit_open()

        vector_state_seed = dict(state)
        graph_state_seed = dict(state)
        graph_state_seed["graph_max_docs"] = graph_budget_docs
        graph_task = None if graph_circuit_open else asyncio.create_task(self.graph_db_retrieval_node(graph_state_seed, runtime))
        vector_state = await asyncio.to_thread(self.vector_db_retrieval_node, vector_state_seed, runtime)
        vector_docs = list(vector_state.get("vector_db_results") or [])
        vector_selected_docs = list(vector_state.get("retrieved_docs") or [])
        vector_confidence = float(vector_state.get("vector_confidence") or 0.0)
        state["vector_db_results"] = vector_docs
        state["vector_retrieval_stats"] = dict(vector_state.get("vector_retrieval_stats") or {})
        should_skip_graph = graph_circuit_open or (
            self.enable_conditional_graph and
            len(vector_docs) >= max(self.conditional_graph_min_vector_docs, max_docs) and
            vector_confidence >= self.conditional_graph_confidence_threshold
        )
        if should_skip_graph:
            graph_docs = []
            state["graph_db_results"] = []
            if graph_task is not None and not graph_task.done():
                graph_task.cancel()
                try:
                    await graph_task
                except asyncio.CancelledError:
                    pass
            if graph_circuit_open:
                self.logger.warning("图检索熔断开启，当前请求跳过图检索")
            else:
                self.logger.info(f"满足条件跳过图检索: vector_docs={len(vector_docs)}, max_docs={max_docs}")
        else:
            graph_state = await graph_task
            graph_docs = list(graph_state.get("graph_db_results") or [])
            state["graph_db_results"] = graph_docs
            state["graph_retrieval_stats"] = dict(graph_state.get("graph_retrieval_stats") or {})

        query_text = state.get("original_question", "")
        merged_docs, rerank_stats = self._merge_retrieved_docs(vector_docs, graph_docs, max_docs, query_text)
        rerank_fallback_rate = 0.0
        if self.semantic_rerank_total_count > 0:
            rerank_fallback_rate = round(self.semantic_rerank_fallback_count / self.semantic_rerank_total_count, 4)
        metric_status = self._emit_semantic_rerank_metrics(rerank_stats)
        state["retrieved_docs"] = merged_docs
        state["vector_db_results"] = vector_docs
        state["graph_db_results"] = graph_docs
        state["retrieval_fusion_stats"] = {
            "vector_docs": len(vector_docs),
            "graph_docs": len(graph_docs),
            "merged_docs": len(merged_docs),
            "rrf_k": self.rrf_k,
            "mmr_lambda": self.mmr_lambda,
            "graph_skipped": should_skip_graph,
            "graph_circuit_open": graph_circuit_open,
            "graph_budget_docs": graph_budget_docs,
            "vector_confidence": vector_confidence,
            "vector_confidence_threshold": self.conditional_graph_confidence_threshold,
            "vector_selected_docs": len(vector_selected_docs),
            "semantic_rerank": rerank_stats,
            "semantic_rerank_fallback_total": self.semantic_rerank_fallback_count,
            "semantic_rerank_total": self.semantic_rerank_total_count,
            "semantic_rerank_fallback_rate": rerank_fallback_rate,
            "semantic_rerank_metrics": metric_status
        }

        self.logger.info(
            f"融合检索完成: vector={len(vector_docs)}, graph={len(graph_docs)}, merged={len(merged_docs)}"
        )
        return state

    async def graph_db_retrieval_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        """图数据库检索节点

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态
        """
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: GRAPH_DB_RETRIEVAL - 图数据库检索")

        # 获取查询相关信息
        original_question = state.get("original_question", "")
        subquestions = state.get("subquestions", [])
        query_text = original_question if original_question else (subquestions[0] if subquestions else "")

        self.logger.info(f"执行图数据库检索，查询: {query_text}")

        try:
            started_at = time.perf_counter()
            context = runtime.context
            max_docs = context.max_retrieval_docs if context else 3
            graph_budget_docs = state.get("graph_max_docs")
            if isinstance(graph_budget_docs, int) and graph_budget_docs > 0:
                max_docs = graph_budget_docs
            cache_key = self._build_retrieval_cache_key(
                prefix="graph",
                query_text=query_text,
                subquestions=[],
                max_docs=max_docs,
                extra=getattr(self.lightrag_storage, "workspace", "default")
            )
            cached_graph = await self._get_cached_item_async(
                self._graph_retrieval_cache,
                cache_key,
                lambda payload: self._deserialize_retrieved_docs(payload.get("docs") or [])
            )
            if cached_graph is not None:
                graph_docs = self._clone_retrieved_docs(cached_graph)
                state["retrieved_docs"] = graph_docs
                state["graph_db_results"] = graph_docs
                state["graph_retrieval_stats"] = {
                    "cache_hit": True,
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                    "retrieved_count": len(graph_docs),
                    "timeout_seconds": self.graph_query_timeout_seconds,
                    "budget_docs": max_docs
                }
                self._record_graph_success(state["graph_retrieval_stats"]["duration_ms"])
                self.logger.info("图数据库检索命中本地缓存")
                return state


            # 执行图数据库检索 - 使用global模式进行图检索
            result = await asyncio.wait_for(
                self.lightrag_storage.query(
                    query=query_text,
                    mode="hybrid",
                    only_need_prompt=True
                ),
                timeout=self.graph_query_timeout_seconds
            )

            # self.logger.info(f"图数据库检索原始结果: {result}")

            # 处理检索结果,只保留Document Chunks部分
            if result and len(result.strip()) > 0:
                # 提取Document Chunks部分
                dc_marker = "-----Document Chunks(DC)-----"
                if dc_marker in result:
                    # 只保留DC标记之后的内容
                    result = result.split(dc_marker, 1)[1].strip()
                    #self.logger.info(f"提取Document Chunks后的结果: {result}")

                graph_docs = self._split_graph_result_to_docs(result, max_docs)

                state["retrieved_docs"] = graph_docs
                state["graph_db_results"] = graph_docs
                await self._set_cached_item_async(
                    self._graph_retrieval_cache,
                    cache_key,
                    self._clone_retrieved_docs(graph_docs),
                    lambda docs: {"docs": self._serialize_retrieved_docs(docs)}
                )

                self.logger.info(f"图数据库检索成功，结果长度: {len(result)}")
                state["graph_retrieval_stats"] = {
                    "cache_hit": False,
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                    "retrieved_count": len(graph_docs),
                    "timeout_seconds": self.graph_query_timeout_seconds,
                    "budget_docs": max_docs
                }
                self._record_graph_success(state["graph_retrieval_stats"]["duration_ms"])
            else:
                self.logger.warning("图数据库检索未返回结果")
                state["retrieved_docs"] = []
                state["graph_db_results"] = []
                await self._set_cached_item_async(
                    self._graph_retrieval_cache,
                    cache_key,
                    [],
                    lambda docs: {"docs": self._serialize_retrieved_docs(docs)}
                )
                state["graph_retrieval_stats"] = {
                    "cache_hit": False,
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                    "retrieved_count": 0,
                    "timeout_seconds": self.graph_query_timeout_seconds,
                    "budget_docs": max_docs
                }
                self._record_graph_success(state["graph_retrieval_stats"]["duration_ms"])

        except asyncio.TimeoutError:
            self.logger.warning(f"图数据库检索超时，已降级为空结果，timeout={self.graph_query_timeout_seconds}s")
            state["retrieved_docs"] = []
            state["graph_db_results"] = []
            state["graph_retrieval_stats"] = {
                "cache_hit": False,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "retrieved_count": 0,
                "timeout_seconds": self.graph_query_timeout_seconds,
                "timed_out": True
            }
            self._record_graph_failure()
        except Exception as e:
            self.logger.error(f"图数据库检索失败: {e}")
            # 检索失败时设置空结果
            state["retrieved_docs"] = []
            state["graph_db_results"] = []
            state["graph_retrieval_stats"] = {
                "cache_hit": False,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "retrieved_count": 0,
                "timeout_seconds": self.graph_query_timeout_seconds,
                "error": str(e)
            }
            self._record_graph_failure()

        return state



    def generate_answer_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        """生成答案节点

        基于检索到的文档和用户问题，调用LLM生成最终答案。

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态，包含生成的答案和更新的messages
        """
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: GENERATE_ANSWER - 生成答案")

        # 获取原始问题
        original_question = state.get("original_question", "")
        retrieved_docs = state.get("retrieved_docs", [])
        
        self.logger.info(f"回答问题: {original_question}")
        self.logger.info(f"可用文档数量: {len(retrieved_docs)}")

        try:
            # 准备文档内容
            documents_text = ""
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs):
                    source = doc.metadata.get("document_name", f"文档{i+1}")
                    documents_text += f"\n[文档 {i+1} - {source}]:\n{doc.page_content}\n"
            else:
                documents_text = "暂无检索到的相关文档。"

            # 获取答案生成提示词
            prompt_template = RAGGraphPrompts.get_answer_generation_prompt()
            prompt = prompt_template.format(
                question=original_question,
                documents=documents_text,
                doc_count=len(retrieved_docs)
            )
            memory_guidance = self._extract_memory_guidance(state)
            if memory_guidance:
                prompt = f"{prompt}\n\n【用户记忆】\n{memory_guidance}\n"

            # 调用LLM生成答案
            answer_result = self.llm.invoke(prompt)
            answer_content = answer_result.content
            
            # 提取文档来源信息
            sources = []
            for i, doc in enumerate(retrieved_docs):
                retrieval_source = doc.metadata.get("source", "vector")
                
                if retrieval_source == "lightrag_graph":
                    content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                else:
                    content_preview = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                chart_image_url = doc.metadata.get("chart_image_url")
                if chart_image_url:
                    try:
                        parsed = urlparse(chart_image_url.strip().strip("`"))
                        host = parsed.netloc or ""
                        if "aliyuncs.com" in host:
                            path = (parsed.path or "").lstrip("/")
                            bucket = host.split(".")[0] if host else None
                            if not bucket or bucket.startswith("oss"):
                                if "/" in path:
                                    bucket, path = path.split("/", 1)
                            key = unquote(path)
                            if bucket and key:
                                presign = get_presigned_url_for_download(bucket=bucket, key=key)
                                if presign and presign.get("url"):
                                    chart_image_url = presign.get("url")
                    except Exception:
                        pass

                source_info = {
                    "index": i + 1,
                    "document_name": doc.metadata.get("document_name", f"文档{i+1}"),
                    "content": content_preview,
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "retrieval_mode": retrieval_source,
                    "content_length": len(doc.page_content),
                    "chunk_type": doc.metadata.get("chunk_type", "text"),
                    "chart_id": doc.metadata.get("chart_id"),
                    "chart_image_url": chart_image_url,
                    "section_title": doc.metadata.get("section_title"),
                    "page_number": doc.metadata.get("page_number")
                }
                sources.append(source_info)

            fallback_docs = list(retrieved_docs or [])
            if not fallback_docs:
                fallback_docs = list(state.get("vector_db_results") or [])

            answer_content = self._repair_contact_answer_if_needed(
                original_question,
                answer_content,
                fallback_docs
            )
            answer_content = self._apply_medical_safety_notice(
                original_question,
                answer_content,
                force_medical=False
            )

            # 更新状态
            state["final_answer"] = answer_content
            state["answer_sources"] = sources
            state["messages"] = [answer_result]

            self.logger.info(f"答案生成成功，包含 {len(sources)} 个来源")

        except Exception as e:
            self.logger.error(f"答案生成失败: {e}")
            error_answer = "抱歉，在生成答案时遇到了技术问题。请稍后重试。"
            state["final_answer"] = error_answer
            state["answer_sources"] = []

        return state

    def direct_answer_node(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        """直接回答节点（支持短期对话记忆）

        对于不需要检索的常规问题，直接使用LLM生成答案，
        并结合当前会话内的对话历史（短期记忆）进行回答。

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态
        """
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: DIRECT_ANSWER - 直接回答")

        all_messages = state.get("messages", [])
        if not all_messages:
            self.logger.warning("没有可用的消息")
            state["final_answer"] = "抱歉，我没有收到您的问题。"
            return state

        latest_message = all_messages[-1]
        user_question = latest_message.content if hasattr(latest_message, "content") else str(latest_message)
        self.logger.info(f"用户问题: {user_question}")

        # 构建对话历史
        try:
            trimmed_messages = all_messages[-20:] if len(all_messages) > 20 else all_messages
            history_lines = []
            for msg in trimmed_messages:
                role = getattr(msg, "type", "user")
                content = msg.content if hasattr(msg, "content") else str(msg)
                if not content:
                    continue
                prefix = "用户" if role in ["human", "user"] else "助手" if role in ["ai", "assistant"] else "系统"
                history_lines.append(f"{prefix}: {content}")

            conversation_history = "\n".join(history_lines)
            if not conversation_history.strip():
                conversation_history = "（当前是本次对话的第一条消息）"

        except Exception as e:
            self.logger.warning(f"构建对话历史失败: {e}")
            conversation_history = "（无法获取对话历史）"

        # 生成回答
        try:
            prompt_template = RAGGraphPrompts.get_direct_answer_prompt()
            prompt = prompt_template.format(
                question=user_question,
                conversation_history=conversation_history
            )
            response = self.llm.invoke(prompt)
            answer = response.content
            answer = self._apply_medical_safety_notice(user_question, answer, force_medical=False)

            state["final_answer"] = answer
            state["messages"] = [AIMessage(content=answer)]
            self.logger.info("直接回答生成成功")

        except Exception as e:
            self.logger.error(f"直接回答失败: {e}")
            error_answer = "抱歉，我在处理您的问题时遇到了问题。请稍后重试。"
            state["final_answer"] = error_answer
            state["messages"] = [AIMessage(content=error_answer)]

        return state

    # ====================记忆功能版本（已注释） ====================
    """
    def direct_answer_node_with_memory(self, state: RAGGraphState, runtime: Runtime[RAGContext]) -> RAGGraphState:
        \"\"\"直接回答节点，集成langmem记忆管理（已弃用）

        对于不需要检索的常规问题，使用create_react_agent模式，
        将记忆管理工具集成到agent中，让LLM自主决定何时使用记忆功能。
        适用于一般性问题、闲聊、简单计算等场景。

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            更新后的状态，包含生成的答案
        \"\"\"
        self.logger.info("=" * 50)
        self.logger.info("[RAG Graph] 节点: DIRECT_ANSWER - 直接回答（集成记忆管理）")

        # 手动裁剪消息，只保留最新的20条
        all_messages = state.get("messages", [])
        if len(all_messages) > 20:
            trimmed_messages = all_messages[-20:]
            self.logger.info(f"消息裁剪: {len(all_messages)} -> {len(trimmed_messages)} 条")
        else:
            trimmed_messages = all_messages

        # 获取最新的用户消息
        latest_message = trimmed_messages[-1]

        # 获取记忆管理agent的提示词
        prompt = RAGGraphPrompts.get_direct_answer_memory_prompt()

        self.logger.info("创建带记忆功能的React Agent...")

        # 创建记忆管理工具 - 使用用户特定的命名空间
        # 获取用户ID，确保记忆隔离
        context = runtime.context
        user_id = context.user_id
        user_namespace = ("memories", user_id)

        tools = [
            # 记忆管理工具 - 可以创建、更新、删除记忆
            # 使用用户特定的命名空间实现数据隔离
            create_manage_memory_tool(namespace=user_namespace, store=self.memory_store),
            # 记忆搜索工具 - 可以搜索相关记忆
            create_search_memory_tool(namespace=user_namespace, store=self.memory_store),
        ]

        # 创建react agent
        agent = create_react_agent(
            self.llm,
            tools=tools,
            prompt=prompt,
            # checkpointer=self.checkpointer,
            # store=self.memory_store
        )

        # 准备agent输入 - 只传递最新的用户消息
        agent_input = {
            "messages": [latest_message]
        }

        # 获取配置信息（包含用户ID和会话ID）
        context = runtime.context
        config = context.get_langgraph_config()

        # 调用agent处理
        self.logger.info("调用React Agent处理用户问题...")
        agent_result = agent.invoke(agent_input, config=config)

        # 获取最终回答 - 使用最新的消息
        agent_messages = agent_result.get("messages", [])
        if agent_messages:
            # 直接使用最后一条消息作为最终答案
            final_answer = agent_messages[-1].content

            self.logger.info("React Agent回答生成成功")

            # 更新状态 - 只添加最新的agent消息
            latest_agent_message = agent_messages[-1]
            state["messages"] = [latest_agent_message]
            state["final_answer"] = final_answer

        return state
    """

    # ==================== 路由函数 ====================

    def route_retrieval_needed(self, state: RAGGraphState) -> str:
        """路由：是否需要检索

        Args:
            state: 当前状态

        Returns:
            路由目标
        """
        need_retrieval = state.get("need_retrieval", False)
        self.logger.info(f"[RAG Graph] 路由检查: need_retrieval = {need_retrieval}")

        if need_retrieval:
            self.logger.info("路由决策: 需要检索 -> check_tool_needed")
            return "need_retrieval"
        else:
            self.logger.info("路由决策: 无需检索 -> check_tool_needed")
            return "no_retrieval"

    def route_tool_needed(self, state: RAGGraphState) -> str:
        """路由：是否需要工具

        Args:
            state: 当前状态

        Returns:
            路由目标
        """
        need_tool = state.get("need_tool", False)
        self.logger.info(f"[RAG Graph] 路由检查: need_tool = {need_tool}")

        if need_tool:
            self.logger.info("路由决策: 需要工具 -> tool_calling")
            return "need_tool"
        else:
            self.logger.info("路由决策: 不需要工具 -> check_retrieval_needed")
            return "no_tool"

    def route_question_type(self, state: RAGGraphState) -> str:
        """路由：检索类型分类

        根据retrieval_mode决定使用哪种检索方式

        Args:
            state: 当前状态

        Returns:
            路由目标
        """
        retrieval_mode = state["retrieval_mode"]

        if retrieval_mode == RetrievalMode.VECTOR_ONLY:
            return "vector_db"
        elif retrieval_mode == RetrievalMode.HYBRID:
            return "hybrid_db"
        elif retrieval_mode == RetrievalMode.GRAPH_ONLY:
            return "graph_db"
        elif retrieval_mode == RetrievalMode.AUTO:
            # AUTO模式默认使用向量检索，可以根据需要扩展智能判断逻辑
            return "vector_db"
        else:
            # 默认使用向量数据库
            return "vector_db"
