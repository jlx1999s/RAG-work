import json
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import redis as redis_sync
from sqlalchemy import or_, inspect
from sqlalchemy.exc import SQLAlchemyError
from backend.config.database import DatabaseFactory
from backend.model.memory import UserMemoryProfile, UserMemoryEvent
from backend.config.log import get_logger

logger = get_logger(__name__)

MEMORY_CACHE_PREFIX = os.getenv("RAG_MEMORY_CACHE_PREFIX", "rag:memory")
MEMORY_CACHE_TTL_SECONDS = int(os.getenv("RAG_MEMORY_CACHE_TTL_SECONDS", "600"))
MEMORY_PROFILE_LIMIT = int(os.getenv("RAG_MEMORY_PROFILE_LIMIT", "12"))
MEMORY_EVENT_LIMIT = int(os.getenv("RAG_MEMORY_EVENT_LIMIT", "8"))
MEMORY_EVENT_DAYS = int(os.getenv("RAG_MEMORY_EVENT_DAYS", "30"))
MEMORY_METRICS_PREFIX = os.getenv("RAG_MEMORY_METRICS_PREFIX", "rag:metrics:memory")
MEMORY_WRITE_MIN_CHARS = int(os.getenv("RAG_MEMORY_WRITE_MIN_CHARS", "6"))
MEMORY_WRITE_MIN_IMPORTANCE = float(os.getenv("RAG_MEMORY_WRITE_MIN_IMPORTANCE", "0.25"))
MEMORY_DEDUP_HOURS = int(os.getenv("RAG_MEMORY_DEDUP_HOURS", "48"))
MEMORY_VECTOR_ENABLED = str(os.getenv("RAG_MEMORY_VECTOR_ENABLED", "false")).strip().lower() in {"1", "true", "yes", "on"}
MEMORY_VECTOR_COLLECTION_PREFIX = os.getenv("RAG_MEMORY_VECTOR_COLLECTION_PREFIX", "memory_events")
MEMORY_VECTOR_TOPK = int(os.getenv("RAG_MEMORY_VECTOR_TOPK", "8"))
MEMORY_PROFILE_COMPLIANCE_ENABLED = str(os.getenv("RAG_MEMORY_PROFILE_COMPLIANCE_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
MEMORY_EVENT_RELEVANCE_ENABLED = str(os.getenv("RAG_MEMORY_EVENT_RELEVANCE_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
MEMORY_EVENT_MIN_RELEVANCE = float(os.getenv("RAG_MEMORY_EVENT_MIN_RELEVANCE", "0.16"))
MEMORY_EVENT_MIN_SCORE = float(os.getenv("RAG_MEMORY_EVENT_MIN_SCORE", "0.22"))
MEMORY_EVENT_TOPIC_MAX = int(os.getenv("RAG_MEMORY_EVENT_TOPIC_MAX", "2"))
MEMORY_EVENT_CANDIDATE_MULTIPLIER = int(os.getenv("RAG_MEMORY_EVENT_CANDIDATE_MULTIPLIER", "5"))
MEMORY_EVENT_TOKEN_BUDGET = int(os.getenv("RAG_MEMORY_EVENT_TOKEN_BUDGET", "420"))
_sync_redis_client = None
_memory_tables_ready = False
_memory_vector_embedding_model = None
_memory_vector_storage_cache: Dict[str, Any] = {}

PROFILE_ALLOWED_KEYS = {
    "response_language",
    "response_style",
    "response_format",
    "user_name",
    "age",
    "gender",
    "conditions",
    "allergies",
    "smoking_status",
    "alcohol_status",
    "salt_intake",
}

PROFILE_TTL_DAYS = {
    "response_language": 365,
    "response_style": 365,
    "response_format": 365,
    "user_name": 365,
    "age": 365,
    "gender": 365,
    "conditions": 180,
    "allergies": 180,
    "smoking_status": 180,
    "alcohol_status": 180,
    "salt_intake": 180,
}


def get_sync_redis_client():
    global _sync_redis_client
    if _sync_redis_client is not None:
        return _sync_redis_client
    try:
        _sync_redis_client = redis_sync.Redis(
            host=os.getenv("REDIS_HOST", "127.0.0.1"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=True,
            socket_connect_timeout=2,
            socket_keepalive=True
        )
    except Exception:
        _sync_redis_client = None
    return _sync_redis_client


def _profile_cache_key(user_id: str, collection_id: str) -> str:
    return f"{MEMORY_CACHE_PREFIX}:profile:{user_id}:{collection_id}"


def _event_cache_key(user_id: str, collection_id: str) -> str:
    return f"{MEMORY_CACHE_PREFIX}:event:{user_id}:{collection_id}"


def _normalize_id(value: Any, default: str = "default") -> str:
    text = str(value or "").strip()
    return text if text else default


def _ensure_memory_tables() -> bool:
    global _memory_tables_ready
    if _memory_tables_ready:
        return True
    try:
        engine = DatabaseFactory.get_engine()
        inspector = inspect(engine)
        profile_exists = inspector.has_table(UserMemoryProfile.__tablename__)
        event_exists = inspector.has_table(UserMemoryEvent.__tablename__)
        if not profile_exists:
            UserMemoryProfile.__table__.create(bind=engine, checkfirst=True)
        if not event_exists:
            UserMemoryEvent.__table__.create(bind=engine, checkfirst=True)
        _memory_tables_ready = True
        return True
    except SQLAlchemyError as exc:
        logger.error(f"检查或创建记忆表失败: {exc}")
        return False


def _extract_profile_candidates(user_text: str) -> Dict[str, str]:
    text = (user_text or "").strip()
    if not text:
        return {}
    profile: Dict[str, str] = {}
    lowered = text.lower()
    if "请用中文" in text or "中文回答" in text:
        profile["response_language"] = "zh"
    elif "请用英文" in text or "英文回答" in text or "in english" in lowered:
        profile["response_language"] = "en"
    if "简洁" in text or "简短" in text:
        profile["response_style"] = "concise"
    elif "详细" in text or "展开" in text:
        profile["response_style"] = "detailed"
    if "markdown" in lowered:
        profile["response_format"] = "markdown"
    elif "纯文本" in text:
        profile["response_format"] = "plain_text"
    name_match = re.search(r"(?:我叫|我是)\s*([^\s，。,.!?！？]{2,20})", text)
    if name_match:
        profile["user_name"] = name_match.group(1).strip()

    age_value = None
    age_match = re.search(r"(?:年龄|age)\s*[:：=]?\s*(\d{1,3})", text, re.IGNORECASE)
    if age_match:
        age_value = age_match.group(1)
    if age_value is None:
        age_match = re.search(r"(?:我|本人|自己).*?(\d{1,3})\s*岁", text)
        if age_match:
            age_value = age_match.group(1)
    if age_value is not None:
        try:
            age_int = int(age_value)
            if 1 <= age_int <= 120:
                profile["age"] = str(age_int)
        except Exception:
            pass

    gender = None
    if re.search(r"(?:我是|性别|gender)\s*[:：=]?\s*(?:男|男性|女|女性)", text, re.IGNORECASE):
        if "女" in text and "男" not in text:
            gender = "female"
        elif "男" in text and "女" not in text:
            gender = "male"
    else:
        if re.search(r"(?:我|本人).*(?:男生|男的|男性)", text):
            gender = "male"
        elif re.search(r"(?:我|本人).*(?:女生|女的|女性)", text):
            gender = "female"
    if gender:
        profile["gender"] = gender

    condition_candidates = {
        "高血压": ["高血压", "血压高", "hypertension"],
        "糖尿病": ["糖尿病", "diabetes"],
        "高脂血症": ["高脂血症", "血脂高", "hyperlipidemia"],
        "冠心病": ["冠心病", "心绞痛", "冠状动脉粥样硬化", "cad"],
        "脑卒中": ["脑卒中", "中风", "stroke"],
        "慢性肾病": ["慢性肾病", "肾功能不全", "ckd"],
        "哮喘": ["哮喘", "asthma"],
    }
    if re.search(r"(?:我|本人|既往史|病史|确诊|诊断|患有|有)\s*", text):
        found_conditions = []
        for canonical, keywords in condition_candidates.items():
            for kw in keywords:
                if kw and kw.lower() in lowered:
                    found_conditions.append(canonical)
                    break
        if found_conditions:
            profile["conditions"] = json.dumps(sorted(set(found_conditions)), ensure_ascii=False)

    allergy_tokens = []
    allergy_match = re.search(r"(?:过敏|allergy)\s*(?:史)?\s*[:：=]?\s*([^。；;\\n]{1,80})", text, re.IGNORECASE)
    if allergy_match:
        allergy_text = allergy_match.group(1).strip()
        if allergy_text and not re.search(r"(?:没有|无|否认|not)", allergy_text, re.IGNORECASE):
            allergy_tokens.append(allergy_text)
    if allergy_tokens:
        profile["allergies"] = json.dumps(sorted(set(allergy_tokens)), ensure_ascii=False)

    if re.search(r"(?:不吸烟|从不吸烟|已戒烟)", text):
        profile["smoking_status"] = "no"
    elif re.search(r"(?:吸烟|抽烟|smoking)", text, re.IGNORECASE) and not re.search(r"(?:不吸烟|从不吸烟|已戒烟)", text):
        profile["smoking_status"] = "yes"

    if re.search(r"(?:不饮酒|从不饮酒|戒酒)", text):
        profile["alcohol_status"] = "no"
    elif re.search(r"(?:饮酒|喝酒|alcohol)", text, re.IGNORECASE) and not re.search(r"(?:不饮酒|从不饮酒|戒酒)", text):
        profile["alcohol_status"] = "yes"

    if re.search(r"(?:低盐|少盐)", text):
        profile["salt_intake"] = "low"
    elif re.search(r"(?:高盐|咸)", text):
        profile["salt_intake"] = "high"

    return profile


def _safe_json_loads(value: Any, default: Any):
    try:
        return json.loads(value)
    except Exception:
        return default


def _collect_query_tokens(text: str) -> set[str]:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", " ", (text or "").lower())
    tokens = [token.strip() for token in cleaned.split() if token.strip()]
    token_set: set[str] = set(tokens)
    if text:
        chinese_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        token_set.update(chinese_chunks)
    return token_set


def _collect_content_tokens(text: str) -> set[str]:
    raw = (text or "").lower()
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", " ", raw)
    tokens = [token.strip() for token in cleaned.split() if token.strip()]
    token_set: set[str] = set(tokens)
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", raw)
    token_set.update(chinese_chunks)
    for chunk in chinese_chunks:
        if len(chunk) <= 2:
            continue
        for idx in range(0, len(chunk) - 1):
            token_set.add(chunk[idx:idx + 2])
        if len(chunk) >= 3:
            for idx in range(0, len(chunk) - 2):
                token_set.add(chunk[idx:idx + 3])
    return token_set


def _score_event_relevance(query_tokens: set[str], content: str) -> float:
    if not query_tokens:
        return 0.0
    content_tokens = _collect_content_tokens(content)
    if not content_tokens:
        return 0.0
    overlap = len(query_tokens & content_tokens)
    return overlap / max(1, len(query_tokens))


def _score_event_recency(created_at_text: str | None) -> float:
    if not created_at_text:
        return 0.0
    try:
        dt = datetime.fromisoformat(str(created_at_text).replace("Z", "+00:00"))
    except Exception:
        return 0.0
    now = datetime.utcnow().replace(tzinfo=dt.tzinfo)
    delta_hours = max(0.0, (now - dt).total_seconds() / 3600.0)
    return max(0.0, 1.0 - min(delta_hours / 168.0, 1.0))


def _rank_events(
    events: List[Dict[str, Any]],
    query_text: str,
    limit: int
) -> List[Dict[str, Any]]:
    if not events:
        return []
    query_tokens = _collect_query_tokens(query_text)
    scored: List[Dict[str, Any]] = []
    for event in events:
        relevance = _score_event_relevance(query_tokens, str(event.get("content") or ""))
        recency = _score_event_recency(event.get("created_at"))
        importance = float(event.get("importance") or 0.0)
        total = relevance * 0.60 + recency * 0.20 + importance * 0.20
        copied = dict(event)
        copied["memory_score"] = round(total, 4)
        copied["memory_relevance"] = round(relevance, 4)
        scored.append(copied)
    scored.sort(key=lambda item: item.get("memory_score", 0.0), reverse=True)
    return scored[:limit]


def _get_memory_vector_storage(collection_id: str):
    global _memory_vector_embedding_model
    if not MEMORY_VECTOR_ENABLED:
        return None
    normalized_collection = _normalize_id(collection_id)
    if normalized_collection in _memory_vector_storage_cache:
        return _memory_vector_storage_cache[normalized_collection]
    try:
        if _memory_vector_embedding_model is None:
            from backend.config.models import initialize_embeddings_model
            _memory_vector_embedding_model = initialize_embeddings_model()
        from backend.rag.storage.milvus_storage import MilvusStorage
        vector_collection_name = f"{MEMORY_VECTOR_COLLECTION_PREFIX}_{normalized_collection}"
        storage = MilvusStorage(
            embedding_function=_memory_vector_embedding_model,
            collection_name=vector_collection_name
        )
        _memory_vector_storage_cache[normalized_collection] = storage
        return storage
    except Exception as exc:
        logger.warning(f"初始化记忆向量存储失败，collection={normalized_collection}, error={exc}")
        return None


def _index_events_to_vector(collection_id: str, events: List[Dict[str, Any]]) -> None:
    if not MEMORY_VECTOR_ENABLED or not events:
        return
    storage = _get_memory_vector_storage(collection_id)
    if not storage:
        return
    try:
        from langchain_core.documents import Document
        documents = []
        ids = []
        for item in events:
            content = str(item.get("content") or "").strip()
            event_id = item.get("id")
            if not content or event_id is None:
                continue
            metadata = {
                "event_id": str(event_id),
                "user_id": str(item.get("user_id") or ""),
                "collection_id": str(item.get("collection_id") or ""),
                "conversation_id": str(item.get("conversation_id") or ""),
                "role": str(item.get("role") or ""),
                "importance": float(item.get("importance") or 0.0),
                "created_at": str(item.get("created_at") or ""),
                "source": "memory_event"
            }
            documents.append(Document(page_content=content, metadata=metadata))
            ids.append(f"memory_event_{event_id}")
        if not documents:
            return
        storage.vector_store.add_documents(documents=documents, ids=ids)
    except Exception as exc:
        logger.warning(f"写入记忆向量索引失败: {exc}")


def _retrieve_vector_event_ids(user_id: str, collection_id: str, query_text: str, top_k: int) -> List[str]:
    if not MEMORY_VECTOR_ENABLED:
        return []
    query = (query_text or "").strip()
    if not query:
        return []
    storage = _get_memory_vector_storage(collection_id)
    if not storage:
        return []
    try:
        docs = storage.hybrid_search(query=query, k=max(4, top_k * 2))
        matched_ids: List[str] = []
        normalized_user = _normalize_id(user_id, "anonymous")
        normalized_collection = _normalize_id(collection_id)
        for doc in docs or []:
            metadata = dict(getattr(doc, "metadata", {}) or {})
            if str(metadata.get("user_id") or "") != normalized_user:
                continue
            if str(metadata.get("collection_id") or "") != normalized_collection:
                continue
            event_id = str(metadata.get("event_id") or "").strip()
            if not event_id or event_id in matched_ids:
                continue
            matched_ids.append(event_id)
            if len(matched_ids) >= top_k:
                break
        return matched_ids
    except Exception as exc:
        logger.warning(f"记忆向量检索失败: {exc}")
        return []


def _apply_vector_boost(events: List[Dict[str, Any]], vector_event_ids: List[str]) -> List[Dict[str, Any]]:
    if not events or not vector_event_ids:
        return events
    rank_map = {event_id: idx for idx, event_id in enumerate(vector_event_ids)}
    boosted = []
    for item in events:
        copied = dict(item)
        event_id = str(item.get("id") or "")
        base_score = float(item.get("memory_score") or 0.0)
        if event_id in rank_map:
            rank = rank_map[event_id]
            boost = max(0.02, 0.12 - 0.015 * rank)
            copied["memory_score"] = round(base_score + boost, 4)
            copied["memory_source"] = "hybrid_vector"
        else:
            copied["memory_source"] = copied.get("memory_source") or "rule"
        boosted.append(copied)
    boosted.sort(key=lambda row: float(row.get("memory_score") or 0.0), reverse=True)
    return boosted


def _retain_vector_ids_in_events(
    events: List[Dict[str, Any]],
    vector_event_ids: List[str],
    redis_client: Any,
    collection_id: str
) -> List[str]:
    if not vector_event_ids:
        return []
    event_ids = {str(item.get("id") or "") for item in events}
    event_ids.discard("")
    retained = [event_id for event_id in vector_event_ids if event_id in event_ids]
    stale_count = len(vector_event_ids) - len(retained)
    if stale_count > 0:
        _inc_metric(redis_client, "vector_struct_mismatch", collection_id)
    if vector_event_ids and not retained:
        _inc_metric(redis_client, "vector_struct_miss_all", collection_id)
    return retained


def _event_topic(content: str) -> str:
    text = str(content or "").lower()
    if re.search(r"(高血压|血压|hypertension|收缩压|舒张压)", text):
        return "hypertension"
    if re.search(r"(糖尿病|血糖|diabetes|hba1c)", text):
        return "diabetes"
    if re.search(r"(胆固醇|血脂|甘油三酯|ldl|hdl)", text):
        return "lipid"
    if re.search(r"(冠心病|脑卒中|心梗|中风|心率)", text):
        return "cardio"
    if re.search(r"(饮食|盐|油|热量|减脂|蛋白|碳水|营养)", text):
        return "diet"
    if re.search(r"(运动|锻炼|跑步|walk|步数|健身)", text):
        return "exercise"
    if re.search(r"(睡眠|失眠|早醒|熬夜)", text):
        return "sleep"
    if re.search(r"(药|用药|服药|剂量|副作用|降压药|胰岛素)", text):
        return "medication"
    return "general"


def _filter_events_by_relevance(events: List[Dict[str, Any]], query_text: str) -> List[Dict[str, Any]]:
    if not events:
        return []
    if not MEMORY_EVENT_RELEVANCE_ENABLED:
        return events
    query_tokens = _collect_query_tokens(query_text)
    if not query_tokens:
        return events
    filtered: List[Dict[str, Any]] = []
    for item in events:
        relevance = float(item.get("memory_relevance") or 0.0)
        score = float(item.get("memory_score") or 0.0)
        source = str(item.get("memory_source") or "")
        if source == "hybrid_vector":
            filtered.append(item)
            continue
        if relevance >= MEMORY_EVENT_MIN_RELEVANCE and score >= MEMORY_EVENT_MIN_SCORE:
            filtered.append(item)
    return filtered


def _select_diverse_events(events: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if not events:
        return []
    normalized_limit = max(1, int(limit or MEMORY_EVENT_LIMIT))
    topic_cap = max(1, int(MEMORY_EVENT_TOPIC_MAX))
    selected: List[Dict[str, Any]] = []
    deferred: List[Dict[str, Any]] = []
    topic_counter: Dict[str, int] = {}
    for item in events:
        if len(selected) >= normalized_limit:
            break
        topic = _event_topic(str(item.get("content") or ""))
        used = topic_counter.get(topic, 0)
        if used >= topic_cap:
            deferred.append(item)
            continue
        selected.append(item)
        topic_counter[topic] = used + 1
    if len(selected) < normalized_limit:
        for item in deferred:
            if len(selected) >= normalized_limit:
                break
            selected.append(item)
    return selected


def _estimate_event_tokens(content: str) -> int:
    text = str(content or "").strip()
    if not text:
        return 0
    return max(1, len(text) // 2)


def _apply_event_budget(events: List[Dict[str, Any]], token_budget: int) -> List[Dict[str, Any]]:
    if not events:
        return []
    budget = max(64, int(token_budget or MEMORY_EVENT_TOKEN_BUDGET))
    kept: List[Dict[str, Any]] = []
    used = 0
    for item in events:
        content_tokens = _estimate_event_tokens(item.get("content"))
        if kept and used + content_tokens > budget:
            continue
        kept.append(item)
        used += content_tokens
    return kept


def _inject_vector_hits(
    ranked_events: List[Dict[str, Any]],
    all_events: List[Dict[str, Any]],
    vector_event_ids: List[str],
    query_text: str,
    candidate_limit: int
) -> List[Dict[str, Any]]:
    if not ranked_events:
        ranked_events = []
    if not vector_event_ids or not all_events:
        return ranked_events[:candidate_limit]
    selected_ids = {str(item.get("id") or "") for item in ranked_events}
    event_map = {str(item.get("id") or ""): item for item in all_events}
    query_tokens = _collect_query_tokens(query_text)
    merged = list(ranked_events)
    for idx, event_id in enumerate(vector_event_ids):
        if len(merged) >= candidate_limit:
            break
        if event_id in selected_ids:
            continue
        raw = event_map.get(event_id)
        if not raw:
            continue
        relevance = _score_event_relevance(query_tokens, str(raw.get("content") or ""))
        recency = _score_event_recency(raw.get("created_at"))
        importance = float(raw.get("importance") or 0.0)
        base_score = relevance * 0.60 + recency * 0.20 + importance * 0.20
        boost = max(0.02, 0.12 - 0.015 * idx)
        copied = dict(raw)
        copied["memory_relevance"] = round(relevance, 4)
        copied["memory_score"] = round(base_score + boost, 4)
        copied["memory_source"] = "hybrid_vector"
        merged.append(copied)
        selected_ids.add(event_id)
    merged.sort(key=lambda row: float(row.get("memory_score") or 0.0), reverse=True)
    return merged[:candidate_limit]


def _rerank_events_stage2(events: List[Dict[str, Any]], query_text: str) -> List[Dict[str, Any]]:
    if not events:
        return []
    query_text_clean = str(query_text or "").strip().lower()
    reranked: List[Dict[str, Any]] = []
    for item in events:
        content = str(item.get("content") or "").strip().lower()
        base_score = float(item.get("memory_score") or 0.0)
        relevance = float(item.get("memory_relevance") or 0.0)
        importance = float(item.get("importance") or 0.0)
        phrase_bonus = 0.0
        if query_text_clean and content and query_text_clean in content:
            phrase_bonus = 0.12
        source_bonus = 0.05 if str(item.get("memory_source") or "") == "hybrid_vector" else 0.0
        rerank_score = base_score * 0.45 + relevance * 0.35 + importance * 0.15 + phrase_bonus + source_bonus
        copied = dict(item)
        copied["memory_rerank_score"] = round(rerank_score, 4)
        reranked.append(copied)
    reranked.sort(key=lambda row: float(row.get("memory_rerank_score") or 0.0), reverse=True)
    return reranked


def _policy_gate_stage3(
    events: List[Dict[str, Any]],
    query_text: str,
    final_limit: int
) -> List[Dict[str, Any]]:
    if not events:
        return []
    filtered = _filter_events_by_relevance(events, query_text)
    diversified = _select_diverse_events(filtered, final_limit)
    budgeted = _apply_event_budget(diversified, MEMORY_EVENT_TOKEN_BUDGET)
    return budgeted[:max(1, int(final_limit or MEMORY_EVENT_LIMIT))]


def _inc_metric(redis_client: Any, metric_key: str, collection_id: str) -> None:
    if not redis_client:
        return
    try:
        minute_bucket = datetime.utcnow().strftime("%Y%m%d%H%M")
        key = f"{MEMORY_METRICS_PREFIX}:{metric_key}:collection:{collection_id}:minute:{minute_bucket}"
        redis_client.incr(key)
        redis_client.expire(key, 7 * 24 * 3600)
    except Exception:
        return


def _inc_metric_by(redis_client: Any, metric_key: str, collection_id: str, amount: int) -> None:
    if not redis_client or amount <= 0:
        return
    try:
        minute_bucket = datetime.utcnow().strftime("%Y%m%d%H%M")
        key = f"{MEMORY_METRICS_PREFIX}:{metric_key}:collection:{collection_id}:minute:{minute_bucket}"
        redis_client.incrby(key, int(amount))
        redis_client.expire(key, 7 * 24 * 3600)
    except Exception:
        return


def _normalize_event_text(text: str) -> str:
    lowered = (text or "").lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^\w\u4e00-\u9fff ]+", "", lowered)
    return lowered


def _compute_event_importance(text: str, profile_candidates: Dict[str, str]) -> float:
    content = (text or "").strip()
    if not content:
        return 0.0
    score = 0.2
    length = len(content)
    if length >= 20:
        score += 0.1
    if length >= 60:
        score += 0.1
    if profile_candidates:
        score += min(0.3, 0.06 * len(profile_candidates))
    if re.search(r"(过敏|病史|确诊|诊断|患有|高血压|糖尿病|冠心病|脑卒中|慢性肾病|哮喘)", content):
        score += 0.2
    if re.search(r"(请用|偏好|习惯|以后|长期|总是)", content):
        score += 0.1
    if re.search(r"(你好|谢谢|收到|在吗|ok|好的)$", content.lower()):
        score -= 0.15
    return max(0.0, min(1.0, score))


def _should_persist_event(text: str, importance: float) -> bool:
    content = (text or "").strip()
    if len(content) < MEMORY_WRITE_MIN_CHARS:
        return False
    if importance < MEMORY_WRITE_MIN_IMPORTANCE:
        return False
    if re.fullmatch(r"(你好|您好|谢谢|收到|好的|ok|嗯+|哈+)[!！。,. ]*", content.lower()):
        return False
    return True


def _evaluate_event_persistence(text: str, importance: float) -> tuple[bool, str]:
    content = (text or "").strip()
    if len(content) < MEMORY_WRITE_MIN_CHARS:
        return False, "too_short"
    if importance < MEMORY_WRITE_MIN_IMPORTANCE:
        return False, "low_importance"
    if re.fullmatch(r"(你好|您好|谢谢|收到|好的|ok|嗯+|哈+)[!！。,. ]*", content.lower()):
        return False, "greeting"
    return True, "ok"


def _is_recent_duplicate_event(
    db: Any,
    user_id: str,
    collection_id: str,
    conversation_id: str,
    role: str,
    normalized_content: str
) -> bool:
    if not normalized_content:
        return False
    since = datetime.utcnow() - timedelta(hours=MEMORY_DEDUP_HOURS)
    rows = db.query(UserMemoryEvent).filter(
        UserMemoryEvent.user_id == user_id,
        UserMemoryEvent.collection_id == collection_id,
        UserMemoryEvent.conversation_id == conversation_id,
        UserMemoryEvent.role == role,
        UserMemoryEvent.created_at >= since
    ).order_by(UserMemoryEvent.created_at.desc()).limit(20).all()
    for row in rows:
        existing = _normalize_event_text(str(row.content or ""))
        if existing and existing == normalized_content:
            return True
    return False


def _should_replace_profile_value(memory_key: str, old_value: str, new_value: str, source_text: str) -> bool:
    if old_value == new_value:
        return True
    text = (source_text or "").strip()
    if memory_key == "age":
        if re.search(r"(?:年龄|age)\s*[:：=]?\s*\d{1,3}", text, re.IGNORECASE):
            return True
        if re.search(r"(?:我|本人|自己).*?\d{1,3}\s*岁", text):
            return True
        return False
    if memory_key == "gender":
        if re.search(r"(?:性别|gender)\s*[:：=]?\s*(?:男|女|男性|女性)", text, re.IGNORECASE):
            return True
        if re.search(r"(?:我是|本人是).*(?:男生|女生|男|女|男性|女性)", text):
            return True
        return False
    return True


def _is_sensitive_value(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if re.search(r"\b1[3-9]\d{9}\b", text):
        return True
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        return True
    if re.search(r"\b\d{15,18}[0-9Xx]\b", text):
        return True
    return False


def _merge_json_array_values(old_value: str, new_value: str) -> str:
    old_items = _safe_json_loads(old_value, [])
    new_items = _safe_json_loads(new_value, [])
    if not isinstance(old_items, list):
        old_items = []
    if not isinstance(new_items, list):
        new_items = []
    merged: List[str] = []
    for token in old_items + new_items:
        cleaned = str(token or "").strip()
        if cleaned and cleaned not in merged:
            merged.append(cleaned)
    return json.dumps(merged[:8], ensure_ascii=False)


def _check_profile_compliance(memory_key: str, memory_value: str) -> tuple[bool, str]:
    key = str(memory_key or "").strip()
    value = str(memory_value or "").strip()
    if not key or not value:
        return False, "empty"
    if MEMORY_PROFILE_COMPLIANCE_ENABLED and key not in PROFILE_ALLOWED_KEYS:
        return False, "key_not_allowed"
    if _is_sensitive_value(value):
        return False, "pii_detected"
    if key == "age":
        if not re.fullmatch(r"\d{1,3}", value):
            return False, "invalid_age"
        age = int(value)
        if not (1 <= age <= 120):
            return False, "invalid_age"
    if key == "gender" and value not in {"male", "female"}:
        return False, "invalid_gender"
    if key == "user_name" and (len(value) > 20 or len(value) < 2):
        return False, "invalid_name"
    return True, "ok"


def _profile_expires_at(memory_key: str) -> Optional[datetime]:
    ttl_days = PROFILE_TTL_DAYS.get(memory_key)
    if not ttl_days:
        return None
    return datetime.utcnow() + timedelta(days=ttl_days)


def get_user_memory_context(
    user_id: str,
    collection_id: str,
    query_text: str = "",
    redis_client: Any = None,
    profile_limit: int = MEMORY_PROFILE_LIMIT,
    event_limit: int = MEMORY_EVENT_LIMIT,
    enforce_relevance: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    if not _ensure_memory_tables():
        return {"profiles": [], "events": []}
    normalized_user = _normalize_id(user_id, "anonymous")
    normalized_collection = _normalize_id(collection_id)
    redis_client = redis_client or get_sync_redis_client()
    candidate_limit = max(
        event_limit,
        min(200, max(1, int(event_limit or MEMORY_EVENT_LIMIT)) * max(1, int(MEMORY_EVENT_CANDIDATE_MULTIPLIER)))
    )

    cached_profile = []
    cached_events = []
    if redis_client:
        try:
            profile_text = redis_client.get(_profile_cache_key(normalized_user, normalized_collection))
            if profile_text:
                cached_profile = _safe_json_loads(profile_text, [])
            event_text = redis_client.get(_event_cache_key(normalized_user, normalized_collection))
            if event_text:
                cached_events = _safe_json_loads(event_text, [])
            if cached_profile or cached_events:
                _inc_metric(redis_client, "read_cache_hit", normalized_collection)
                stage1_candidates = _rank_events(cached_events, query_text, candidate_limit)
                vector_ids = _retrieve_vector_event_ids(
                    user_id=normalized_user,
                    collection_id=normalized_collection,
                    query_text=query_text,
                    top_k=max(candidate_limit, MEMORY_VECTOR_TOPK)
                )
                retained_ids = _retain_vector_ids_in_events(
                    events=stage1_candidates,
                    vector_event_ids=vector_ids,
                    redis_client=redis_client,
                    collection_id=normalized_collection
                )
                stage1_candidates = _apply_vector_boost(stage1_candidates, retained_ids)
                stage1_candidates = _inject_vector_hits(
                    ranked_events=stage1_candidates,
                    all_events=cached_events,
                    vector_event_ids=retained_ids,
                    query_text=query_text,
                    candidate_limit=candidate_limit
                )
                stage2_reranked = _rerank_events_stage2(stage1_candidates, query_text)
                if enforce_relevance:
                    stage3_final = _policy_gate_stage3(stage2_reranked, query_text, event_limit)
                else:
                    stage3_final = stage2_reranked[:event_limit]
                return {"profiles": cached_profile[:profile_limit], "events": stage3_final}
        except Exception as exc:
            logger.warning(f"读取记忆缓存失败: {exc}")
            _inc_metric(redis_client, "read_cache_error", normalized_collection)

    db = None
    profiles: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    try:
        db = DatabaseFactory.create_session()
        now = datetime.utcnow()
        profile_rows = db.query(UserMemoryProfile).filter(
            UserMemoryProfile.user_id == normalized_user,
            UserMemoryProfile.collection_id == normalized_collection,
            UserMemoryProfile.is_deleted == False,
            or_(UserMemoryProfile.expires_at.is_(None), UserMemoryProfile.expires_at >= now)
        ).order_by(UserMemoryProfile.updated_at.desc()).limit(profile_limit).all()
        event_rows = db.query(UserMemoryEvent).filter(
            UserMemoryEvent.user_id == normalized_user,
            UserMemoryEvent.collection_id == normalized_collection,
            or_(UserMemoryEvent.expires_at.is_(None), UserMemoryEvent.expires_at >= now)
        ).order_by(UserMemoryEvent.created_at.desc()).limit(candidate_limit).all()
        profiles = [row.to_dict() for row in profile_rows]
        events = [row.to_dict() for row in event_rows]
        _inc_metric(redis_client, "read_db_hit", normalized_collection)
    except Exception as exc:
        logger.error(f"读取用户记忆失败: {exc}")
        _inc_metric(redis_client, "read_db_error", normalized_collection)
    finally:
        if db:
            db.close()

    if redis_client:
        try:
            redis_client.setex(
                _profile_cache_key(normalized_user, normalized_collection),
                MEMORY_CACHE_TTL_SECONDS,
                json.dumps(profiles, ensure_ascii=False)
            )
            redis_client.setex(
                _event_cache_key(normalized_user, normalized_collection),
                MEMORY_CACHE_TTL_SECONDS,
                json.dumps(events, ensure_ascii=False)
            )
        except Exception as exc:
            logger.warning(f"写入记忆缓存失败: {exc}")
            _inc_metric(redis_client, "write_cache_error", normalized_collection)

    stage1_candidates = _rank_events(events, query_text, candidate_limit)
    vector_event_ids = _retrieve_vector_event_ids(
        user_id=normalized_user,
        collection_id=normalized_collection,
        query_text=query_text,
        top_k=max(candidate_limit, MEMORY_VECTOR_TOPK)
    )
    retained_vector_ids = _retain_vector_ids_in_events(
        events=stage1_candidates,
        vector_event_ids=vector_event_ids,
        redis_client=redis_client,
        collection_id=normalized_collection
    )
    stage1_candidates = _apply_vector_boost(stage1_candidates, retained_vector_ids)
    stage1_candidates = _inject_vector_hits(
        ranked_events=stage1_candidates,
        all_events=events,
        vector_event_ids=retained_vector_ids,
        query_text=query_text,
        candidate_limit=candidate_limit
    )
    stage2_reranked = _rerank_events_stage2(stage1_candidates, query_text)
    if enforce_relevance:
        stage3_final = _policy_gate_stage3(stage2_reranked, query_text, event_limit)
    else:
        stage3_final = stage2_reranked[:event_limit]
    return {"profiles": profiles, "events": stage3_final}


def build_memory_system_prompt(memory_context: Dict[str, List[Dict[str, Any]]]) -> str:
    profiles = memory_context.get("profiles") or []
    events = memory_context.get("events") or []
    profile_lines = []
    for item in profiles[:MEMORY_PROFILE_LIMIT]:
        key = item.get("memory_key")
        value = item.get("memory_value")
        if key and value:
            profile_lines.append(f"- {key}: {value}")
    event_lines = []
    for item in events[:MEMORY_EVENT_LIMIT]:
        role = item.get("role", "user")
        content = str(item.get("content") or "").strip()
        if content:
            event_lines.append(f"- [{role}] {content[:160]}")
    if not profile_lines and not event_lines:
        return ""
    sections = []
    if profile_lines:
        sections.append("用户稳定偏好：\n" + "\n".join(profile_lines))
    if event_lines:
        sections.append("近期关键会话记忆：\n" + "\n".join(event_lines))
    return "以下是系统检索到的用户记忆，仅在与当前问题相关时参考；如果无关请忽略。\n" + "\n\n".join(sections)


def write_user_memory(
    user_id: str,
    collection_id: str,
    conversation_id: str,
    user_message: str,
    assistant_message: Optional[str] = None,
    redis_client: Any = None
) -> None:
    if not _ensure_memory_tables():
        return
    normalized_user = _normalize_id(user_id, "anonymous")
    normalized_collection = _normalize_id(collection_id)
    normalized_conversation = _normalize_id(conversation_id, "default")
    message_text = (user_message or "").strip()
    if not message_text:
        return
    redis_client = redis_client or get_sync_redis_client()

    db = None
    indexed_events: List[Dict[str, Any]] = []
    try:
        db = DatabaseFactory.create_session()
        profile_candidates = _extract_profile_candidates(message_text)
        if not profile_candidates:
            _inc_metric(redis_client, "profile_extract_empty", normalized_collection)
        else:
            _inc_metric_by(redis_client, "profile_extract_candidates", normalized_collection, len(profile_candidates))
        for key, value in profile_candidates.items():
            compliant, reason = _check_profile_compliance(key, value)
            if not compliant:
                _inc_metric(redis_client, f"profile_skip_{reason}", normalized_collection)
                continue
            expires_at_profile = _profile_expires_at(key)
            row = db.query(UserMemoryProfile).filter(
                UserMemoryProfile.user_id == normalized_user,
                UserMemoryProfile.collection_id == normalized_collection,
                UserMemoryProfile.memory_key == key
            ).first()
            if row:
                old_value = str(row.memory_value or "")
                candidate_value = value
                if key in {"conditions", "allergies"}:
                    candidate_value = _merge_json_array_values(old_value, value)
                if _should_replace_profile_value(key, old_value, candidate_value, message_text):
                    row.memory_value = candidate_value
                    if old_value == candidate_value:
                        row.confidence = min(0.98, float(row.confidence or 0.7) + 0.03)
                    else:
                        row.confidence = 0.85
                    row.source = "chat_extractor"
                    row.is_deleted = False
                    row.expires_at = expires_at_profile
                    _inc_metric(redis_client, "profile_update_success", normalized_collection)
                else:
                    row.confidence = max(0.5, float(row.confidence or 0.7) - 0.05)
                    _inc_metric(redis_client, "profile_conflict_skip", normalized_collection)
            else:
                row = UserMemoryProfile(
                    user_id=normalized_user,
                    collection_id=normalized_collection,
                    memory_key=key,
                    memory_value=value,
                    confidence=0.8,
                    source="chat_extractor",
                    expires_at=expires_at_profile
                )
                db.add(row)
                _inc_metric(redis_client, "profile_insert_success", normalized_collection)

        user_importance = _compute_event_importance(message_text, profile_candidates)
        expires_at = datetime.utcnow() + timedelta(days=MEMORY_EVENT_DAYS)
        normalized_user_event = _normalize_event_text(message_text)
        should_user_write, user_skip_reason = _evaluate_event_persistence(message_text, user_importance)
        if should_user_write and not _is_recent_duplicate_event(
            db=db,
            user_id=normalized_user,
            collection_id=normalized_collection,
            conversation_id=normalized_conversation,
            role="user",
            normalized_content=normalized_user_event
        ):
            user_event = UserMemoryEvent(
                user_id=normalized_user,
                conversation_id=normalized_conversation,
                collection_id=normalized_collection,
                role="user",
                content=message_text,
                importance=user_importance,
                source="chat",
                expires_at=expires_at
            )
            db.add(user_event)
            db.flush()
            indexed_events.append(user_event.to_dict())
            _inc_metric(redis_client, "event_user_write_success", normalized_collection)
        elif not should_user_write:
            _inc_metric(redis_client, f"event_user_skip_{user_skip_reason}", normalized_collection)
        else:
            _inc_metric(redis_client, "event_user_skip_duplicate", normalized_collection)

        assistant_text = (assistant_message or "").strip()
        if assistant_text:
            assistant_importance = _compute_event_importance(assistant_text, {})
            normalized_assistant_event = _normalize_event_text(assistant_text[:1000])
            should_assistant_write, assistant_skip_reason = _evaluate_event_persistence(assistant_text, assistant_importance)
            if should_assistant_write and not _is_recent_duplicate_event(
                db=db,
                user_id=normalized_user,
                collection_id=normalized_collection,
                conversation_id=normalized_conversation,
                role="assistant",
                normalized_content=normalized_assistant_event
            ):
                assistant_event = UserMemoryEvent(
                    user_id=normalized_user,
                    conversation_id=normalized_conversation,
                    collection_id=normalized_collection,
                    role="assistant",
                    content=assistant_text[:1000],
                    importance=assistant_importance,
                    source="chat",
                    expires_at=expires_at
                )
                db.add(assistant_event)
                db.flush()
                indexed_events.append(assistant_event.to_dict())
                _inc_metric(redis_client, "event_assistant_write_success", normalized_collection)
            elif not should_assistant_write:
                _inc_metric(redis_client, f"event_assistant_skip_{assistant_skip_reason}", normalized_collection)
            else:
                _inc_metric(redis_client, "event_assistant_skip_duplicate", normalized_collection)

        db.commit()
        _index_events_to_vector(collection_id=normalized_collection, events=indexed_events)
        _inc_metric(redis_client, "write_db_success", normalized_collection)
    except Exception as exc:
        logger.error(f"写入用户记忆失败: {exc}")
        _inc_metric(redis_client, "write_db_error", normalized_collection)
        if db:
            db.rollback()
    finally:
        if db:
            db.close()

    if redis_client:
        try:
            redis_client.delete(_profile_cache_key(normalized_user, normalized_collection))
            redis_client.delete(_event_cache_key(normalized_user, normalized_collection))
        except Exception as exc:
            logger.warning(f"清理记忆缓存失败: {exc}")


def list_user_memories(user_id: str, collection_id: str, profile_limit: int = 50, event_limit: int = 50) -> Dict[str, Any]:
    normalized_user = _normalize_id(user_id, "anonymous")
    normalized_collection = _normalize_id(collection_id)
    memory_context = get_user_memory_context(
        user_id=normalized_user,
        collection_id=normalized_collection,
        query_text="",
        profile_limit=profile_limit,
        event_limit=event_limit,
        enforce_relevance=False
    )
    return {
        "user_id": normalized_user,
        "collection_id": normalized_collection,
        "profiles": memory_context.get("profiles", []),
        "events": memory_context.get("events", [])
    }


def delete_memory_profile(user_id: str, collection_id: str, memory_key: str) -> bool:
    if not _ensure_memory_tables():
        return False
    normalized_user = _normalize_id(user_id, "anonymous")
    normalized_collection = _normalize_id(collection_id)
    key = str(memory_key or "").strip()
    if not key:
        return False
    db = None
    try:
        db = DatabaseFactory.create_session()
        row = db.query(UserMemoryProfile).filter(
            UserMemoryProfile.user_id == normalized_user,
            UserMemoryProfile.collection_id == normalized_collection,
            UserMemoryProfile.memory_key == key
        ).first()
        if not row:
            return False
        row.is_deleted = True
        db.commit()
    except Exception as exc:
        logger.error(f"删除画像记忆失败: {exc}")
        if db:
            db.rollback()
        return False
    finally:
        if db:
            db.close()
    redis_client = get_sync_redis_client()
    if redis_client:
        try:
            redis_client.delete(_profile_cache_key(normalized_user, normalized_collection))
        except Exception:
            pass
    return True


def delete_memory_events_by_conversation(user_id: str, conversation_id: str, collection_id: str) -> int:
    if not _ensure_memory_tables():
        return 0
    normalized_user = _normalize_id(user_id, "anonymous")
    normalized_collection = _normalize_id(collection_id)
    normalized_conversation = _normalize_id(conversation_id, "default")
    db = None
    deleted_count = 0
    try:
        db = DatabaseFactory.create_session()
        deleted_count = db.query(UserMemoryEvent).filter(
            UserMemoryEvent.user_id == normalized_user,
            UserMemoryEvent.collection_id == normalized_collection,
            UserMemoryEvent.conversation_id == normalized_conversation
        ).delete()
        db.commit()
    except Exception as exc:
        logger.error(f"按会话删除事件记忆失败: {exc}")
        if db:
            db.rollback()
        return 0
    finally:
        if db:
            db.close()
    redis_client = get_sync_redis_client()
    if redis_client:
        try:
            redis_client.delete(_event_cache_key(normalized_user, normalized_collection))
        except Exception:
            pass
    return int(deleted_count)


def delete_all_user_memory(user_id: str, collection_id: str) -> Dict[str, int]:
    if not _ensure_memory_tables():
        return {"profiles": 0, "events": 0}
    normalized_user = _normalize_id(user_id, "anonymous")
    normalized_collection = _normalize_id(collection_id)
    db = None
    profile_count = 0
    event_count = 0
    try:
        db = DatabaseFactory.create_session()
        profile_count = db.query(UserMemoryProfile).filter(
            UserMemoryProfile.user_id == normalized_user,
            UserMemoryProfile.collection_id == normalized_collection,
            UserMemoryProfile.is_deleted == False
        ).update({"is_deleted": True}, synchronize_session=False)
        event_count = db.query(UserMemoryEvent).filter(
            UserMemoryEvent.user_id == normalized_user,
            UserMemoryEvent.collection_id == normalized_collection
        ).delete()
        db.commit()
    except Exception as exc:
        logger.error(f"删除用户全部记忆失败: {exc}")
        if db:
            db.rollback()
        return {"profiles": 0, "events": 0}
    finally:
        if db:
            db.close()
    redis_client = get_sync_redis_client()
    if redis_client:
        try:
            redis_client.delete(_profile_cache_key(normalized_user, normalized_collection))
            redis_client.delete(_event_cache_key(normalized_user, normalized_collection))
        except Exception:
            pass
    return {"profiles": int(profile_count), "events": int(event_count)}
