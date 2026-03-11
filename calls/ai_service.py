import time
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
from django.conf import settings

def generate_response(messages: list, model_key: str = None) -> dict:
    from groq import Groq
    key = model_key or settings.ACTIVE_MODEL
    if key not in settings.MODEL_CONFIGS:
        key = list(settings.MODEL_CONFIGS.keys())[0]
    cfg = settings.MODEL_CONFIGS[key]
    client = Groq(api_key=settings.GROQ_API_KEY)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=cfg["model_id"],
        messages=messages,
        temperature=0.3,
        max_tokens=300,
    )
    latency = (time.perf_counter() - t0) * 1000
    usage = resp.usage
    return {
        "content":           resp.choices[0].message.content.strip(),
        "llm_latency_ms":    round(latency, 2),
        "prompt_tokens":     usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens":      usage.total_tokens,
        "model_key":         key,
        "model_id":          cfg["model_id"],
        "provider":          cfg["provider"],
    }
