import logging
import time
import traceback

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from calls.ai_service import generate_response
from calls.models import CallSession, TurnMetrics
from calls.session_cache import get_session, save_session_to_cache
from calls.tasks import score_and_persist_turn
from calls.twilio_service import twiml_gather, twiml_hangup
# retrieve() and build_system_prompt() delegate to the ChromaDB singleton
# initialised once at startup via KnowledgeBaseConfig.ready() — no per-request
# indexing or file I/O occurs here.
from knowledge_base.kb_service import build_system_prompt, retrieve

logger      = logging.getLogger(__name__)
latency_log = logging.getLogger("latency")


# ── Helpers ────────────────────────────────────────────────────────────────

def _twiml_error():
    return HttpResponse(
        twiml_hangup("Sorry, an error occurred."),
        content_type="text/xml",
    )


# ── Views ──────────────────────────────────────────────────────────────────

@csrf_exempt
def answer_call(request):
    session_id = request.GET.get("session_id", "")
    call_sid   = request.POST.get("CallSid", "")
    logger.info("answer_call session_id=%s call_sid=%s", session_id, call_sid)

    session = get_session(session_id)
    if session:
        session.call_sid = call_sid
        session.status   = "in-progress"
        # Persist status change to DB + refresh Redis immediately
        session.save()
        save_session_to_cache(session_id, session)

    return HttpResponse(
        twiml_gather("Hello! I am your AI assistant. How can I help you today?", session_id),
        content_type="text/xml",
    )


@csrf_exempt
def respond_to_speech(request):
    """
    Hot path — must return TwiML as fast as possible.

    Timing budget per turn (target ~6 s total):
        Redis read    ~5 ms       (was MongoDB read ~300 ms)
        KB retrieve   ~50 ms
        Groq LLM     ~300–500 ms
        Redis write   ~5 ms       (was MongoDB write ~300 ms)
        ──────────────────────
        Total view   ~400–600 ms  (scoring/DB write moved to Celery)
    """
    turn_started_at = time.perf_counter()
    try:
        session_id  = request.GET.get("session_id", "")
        user_speech = request.POST.get("SpeechResult", "").strip()

        if not user_speech:
            return HttpResponse(
                twiml_hangup("I didn't hear anything. Goodbye!"),
                content_type="text/xml",
            )

        # ── 1. Fetch session (Redis HIT: ~5 ms) ───────────────────────────
        session = get_session(session_id)
        if not session:
            return HttpResponse(
                twiml_hangup("Session error. Goodbye!"),
                content_type="text/xml",
            )

        # ── 2. KB retrieval + LLM ─────────────────────────────────────────
        retrieved    = retrieve(user_speech, top_k=3)
        context_texts = [r[0] for r in retrieved]
        system_prompt = build_system_prompt(retrieved)

        session.full_transcript.append({"role": "user", "content": user_speech})
        messages = [{"role": "system", "content": system_prompt}] + session.full_transcript

        ai_result     = generate_response(messages, model_key=session.model_key)
        response_text = ai_result.get("content", "")

        # ── 3. Build TurnMetrics (in-memory only at this point) ───────────
        turn = TurnMetrics(
            turn_index        = len(session.turns),
            user_input        = user_speech,
            ai_response       = response_text,
            llm_latency_ms    = ai_result.get("llm_latency_ms"),
            total_latency_ms  = round((time.perf_counter() - turn_started_at) * 1000, 2),
            prompt_tokens     = ai_result.get("prompt_tokens"),
            completion_tokens = ai_result.get("completion_tokens"),
            total_tokens      = ai_result.get("total_tokens"),
            hallucination_score = None,
            faithfulness_score  = None,
            context_used      = context_texts,
            evaluation_notes  = "Evaluation scheduled in Celery.",
        )

        turn_index = turn.turn_index
        session.turns.append(turn)
        session.full_transcript.append({"role": "assistant", "content": response_text})
        session.model_id       = ai_result.get("model")
        session.model_provider = ai_result.get("provider")

        # ── 4. Write to Redis only (~5 ms) — DB write happens in Celery ───
        save_session_to_cache(session_id, session)

        latency_log.info(
            "session=%s turn=%d model=%s llm_ms=%.1f total_ms=%.1f "
            "tokens=%d/%d/%d",
            session_id,
            turn_index,
            ai_result.get("model", "-"),
            turn.llm_latency_ms,
            turn.total_latency_ms,
            ai_result.get("prompt_tokens", 0),
            ai_result.get("completion_tokens", 0),
            ai_result.get("total_tokens", 0),
        )

        # ── 5. Fire Celery task (non-blocking) ────────────────────────────
        score_and_persist_turn.delay(
            session_id,
            turn_index,
            response_text,
            context_texts,
            user_speech,
        )

        # ── 6. Return TwiML immediately ───────────────────────────────────
        end_words = ["bye", "goodbye", "hang up", "that's all", "nothing else", "thank you bye"]
        if any(w in user_speech.lower() for w in end_words):
            return HttpResponse(
                twiml_hangup(f"{response_text} Goodbye!"),
                content_type="text/xml",
            )

        return HttpResponse(twiml_gather(response_text, session_id), content_type="text/xml")

    except Exception:
        logger.exception("Error in respond_to_speech")
        logger.error(traceback.format_exc())
        return _twiml_error()


@csrf_exempt
def call_status(request):
    call_sid = request.POST.get("CallSid", "")
    status   = request.POST.get("CallStatus", "")
    duration = request.POST.get("CallDuration", "0")
    try:
        s = CallSession.objects.get(call_sid=call_sid)
        s.status         = status
        s.call_duration_s = float(duration)
        s.finalize()  # finalize() calls s.save() internally
    except Exception:
        logger.exception("call_status update failed call_sid=%s", call_sid)
    return HttpResponse("OK")
