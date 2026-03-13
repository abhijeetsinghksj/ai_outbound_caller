"""
Redis-backed session cache for CallSession documents.

Hot path (respond_to_speech):
    get_session()           ~5 ms  (Redis HIT)
    save_session_to_cache() ~5 ms  (Redis SET)

Previously these were MongoDB reads/writes: ~200-500 ms each.
MongoDB writes now happen exclusively inside the Celery background task.
"""
import json
import logging
import datetime

import redis as _redis_lib
from django.conf import settings

logger = logging.getLogger(__name__)

# ── Redis connection (lazy singleton) ──────────────────────────────────────

_redis_client = None


def _get_redis() -> _redis_lib.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = _redis_lib.Redis.from_url(
            settings.REDIS_SESSION_URL,
            decode_responses=True,
        )
    return _redis_client


def _cache_key(session_id: str) -> str:
    return f"sess:{session_id}"


# ── Serialisation helpers ──────────────────────────────────────────────────

def _isoformat(dt: datetime.datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _parse_dt(value: str | None) -> datetime.datetime | None:
    return datetime.datetime.fromisoformat(value) if value else None


def session_to_dict(session) -> dict:
    """Convert a CallSession mongoengine document to a JSON-serialisable dict."""
    return {
        "id":                   str(session.id),
        "call_sid":             session.call_sid,
        "to_number":            session.to_number,
        "from_number":          session.from_number,
        "status":               session.status,
        "model_key":            session.model_key,
        "model_id":             session.model_id,
        "model_provider":       session.model_provider,
        "full_transcript":      session.full_transcript,  # list[dict] — already JSON-safe
        "avg_llm_latency_ms":   session.avg_llm_latency_ms,
        "avg_total_latency_ms": session.avg_total_latency_ms,
        "avg_hallucination":    session.avg_hallucination,
        "avg_faithfulness":     session.avg_faithfulness,
        "total_turns":          session.total_turns,
        "call_duration_s":      session.call_duration_s,
        "started_at":           _isoformat(session.started_at),
        "ended_at":             _isoformat(session.ended_at),
        "turns": [
            {
                "turn_index":         t.turn_index,
                "user_input":         t.user_input,
                "ai_response":        t.ai_response,
                "llm_latency_ms":     t.llm_latency_ms,
                "tts_latency_ms":     t.tts_latency_ms,
                "total_latency_ms":   t.total_latency_ms,
                "prompt_tokens":      t.prompt_tokens,
                "completion_tokens":  t.completion_tokens,
                "total_tokens":       t.total_tokens,
                "hallucination_score": t.hallucination_score,
                "faithfulness_score":  t.faithfulness_score,
                "context_used":        list(t.context_used),
                "evaluation_notes":    t.evaluation_notes,
                "timestamp":           _isoformat(t.timestamp),
            }
            for t in session.turns
        ],
    }


def dict_to_session(data: dict):
    """
    Reconstruct a CallSession mongoengine object from a cached dict.

    The returned object has _created=False so that calling .save() on it
    triggers an UPDATE rather than an INSERT.
    """
    from calls.models import CallSession, TurnMetrics
    from bson import ObjectId

    turns = [
        TurnMetrics(
            turn_index        = td["turn_index"],
            user_input        = td.get("user_input", ""),
            ai_response       = td.get("ai_response", ""),
            llm_latency_ms    = td.get("llm_latency_ms", 0.0),
            tts_latency_ms    = td.get("tts_latency_ms", 0.0),
            total_latency_ms  = td.get("total_latency_ms", 0.0),
            prompt_tokens     = td.get("prompt_tokens", 0),
            completion_tokens = td.get("completion_tokens", 0),
            total_tokens      = td.get("total_tokens", 0),
            hallucination_score = td.get("hallucination_score"),
            faithfulness_score  = td.get("faithfulness_score"),
            context_used      = td.get("context_used", []),
            evaluation_notes  = td.get("evaluation_notes", ""),
            timestamp         = _parse_dt(td.get("timestamp")),
        )
        for td in data.get("turns", [])
    ]

    # Build the document via explicit attribute assignment so mongoengine
    # marks every field as "changed" (required for _save_update to work).
    session = CallSession()
    session.id              = ObjectId(data["id"])
    session.call_sid        = data.get("call_sid", "pending")
    session.to_number       = data.get("to_number", "")
    session.from_number     = data.get("from_number")
    session.status          = data.get("status", "initiated")
    session.model_key       = data.get("model_key")
    session.model_id        = data.get("model_id")
    session.model_provider  = data.get("model_provider")
    session.turns           = turns
    session.full_transcript = data.get("full_transcript", [])
    session.avg_llm_latency_ms   = data.get("avg_llm_latency_ms", 0.0)
    session.avg_total_latency_ms = data.get("avg_total_latency_ms", 0.0)
    session.avg_hallucination    = data.get("avg_hallucination")
    session.avg_faithfulness     = data.get("avg_faithfulness")
    session.total_turns     = data.get("total_turns", 0)
    session.call_duration_s = data.get("call_duration_s", 0.0)
    session.started_at      = _parse_dt(data.get("started_at"))
    session.ended_at        = _parse_dt(data.get("ended_at"))

    # Tell mongoengine this document already exists in MongoDB → use UPDATE path.
    session._created = False
    return session


# ── Public API ─────────────────────────────────────────────────────────────

def get_session(session_id: str):
    """
    Return a CallSession object.

    1. Check Redis (O(1), ~5 ms) → HIT: deserialise and return.
    2. Miss: fetch from MongoDB, prime Redis, return.
    """
    r = _get_redis()
    raw = r.get(_cache_key(session_id))

    if raw:
        logger.info("[CACHE] HIT  session=%s", session_id)
        return dict_to_session(json.loads(raw))

    logger.info("[CACHE] MISS session=%s — fetching from MongoDB", session_id)
    from calls.models import CallSession
    try:
        session = CallSession.objects.get(id=session_id)
        save_session_to_cache(session_id, session)
        return session
    except Exception:
        logger.exception("[CACHE] MongoDB fetch failed session=%s", session_id)
        return None


def save_session_to_cache(session_id: str, session) -> None:
    """Serialise CallSession to Redis with TTL. ~5 ms."""
    r = _get_redis()
    data = session_to_dict(session)
    r.setex(
        _cache_key(session_id),
        settings.REDIS_SESSION_TTL,
        json.dumps(data),
    )
    logger.info(
        "[CACHE] SAVED session=%s turns=%d",
        session_id,
        len(session.turns),
    )
