import logging
import os
import re
import time

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
from django.conf import settings

latency_log = logging.getLogger("latency")


def _resolve_model(model_key: str | None) -> tuple[str, dict]:
    key = model_key or settings.ACTIVE_MODEL
    if key not in settings.MODEL_CONFIGS:
        key = list(settings.MODEL_CONFIGS.keys())[0]
    return key, settings.MODEL_CONFIGS[key]


def _sanitize_for_tts(text: str) -> str:
    """Normalize model text for safer, cleaner Twilio TTS playback."""
    if not text:
        return "I'm sorry, I don't have a response right now."

    cleaned = text.replace("\r", " ").replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Remove control characters that can break or degrade TTS behavior.
    cleaned = "".join(ch for ch in cleaned if ch.isprintable())
    return cleaned.strip()


def _request_completion(messages: list, model_id: str):
    """Single place for non-streaming completion request. Easy to swap to streaming later."""
    from groq import Groq

    client = Groq(api_key=settings.GROQ_API_KEY)
    return client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.3,
        max_tokens=300,
    )


def generate_response(messages: list, model_key: str = None) -> dict:
    key, cfg = _resolve_model(model_key)

    t0 = time.perf_counter()
    resp = _request_completion(messages=messages, model_id=cfg["model_id"])
    latency = (time.perf_counter() - t0) * 1000

    content = _sanitize_for_tts(resp.choices[0].message.content)

    latency_log.info(
        "[Groq] model=%s llm_ms=%.1f prompt_tok=%d completion_tok=%d total_tok=%d",
        cfg["model_id"],
        round(latency, 2),
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
        resp.usage.total_tokens,
    )

    return {
        "content": content,
        "llm_latency_ms": round(latency, 2),
        "model": cfg["model_id"],
        "model_key": key,
        "provider": cfg["provider"],
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "total_tokens": resp.usage.total_tokens,
    }
