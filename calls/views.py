import logging
import threading
import time
import traceback

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from calls.ai_service import generate_response
from calls.models import CallSession, TurnMetrics
from calls.twilio_service import twiml_gather, twiml_hangup
from calls.ai_service import generate_response
# retrieve() and build_system_prompt() delegate to the ChromaDB singleton
# initialised once at startup via KnowledgeBaseConfig.ready() — no per-request
# indexing or file I/O occurs here.
from knowledge_base.kb_service import retrieve, build_system_prompt
from evaluation.eval_service import score_response
from knowledge_base.kb_service import build_system_prompt, retrieve

logger = logging.getLogger(__name__)
latency_log = logging.getLogger("latency")


def _get_session(sid):
    try:
        return CallSession.objects.get(id=sid)
    except Exception:
        logger.exception("Failed to load session", extra={"session_id": sid})
        return None


def _score_and_persist_turn(session_id, turn_index, ai_text, context_texts, user_speech):
    """Background hallucination scoring that updates the turn without blocking Twilio response."""
    try:
        eval_result = score_response(ai_text, context_texts, user_speech)
        session = CallSession.objects.get(id=session_id)
        if turn_index >= len(session.turns):
            logger.warning(
                "Skipping eval persistence: turn index out of range",
                extra={"session_id": str(session_id), "turn_index": turn_index},
            )
            return

        session.turns[turn_index].hallucination_score = eval_result.get("hallucination_score")
        session.turns[turn_index].faithfulness_score = eval_result.get("faithfulness_score")
        session.turns[turn_index].evaluation_notes = eval_result.get("notes")
        session.save()
        logger.info("Evaluation persisted", extra={"session_id": str(session_id), "turn_index": turn_index})
    except Exception:
        logger.exception(
            "Background evaluation failed",
            extra={"session_id": str(session_id), "turn_index": turn_index},
        )


@csrf_exempt
def answer_call(request):
    session_id = request.GET.get("session_id", "")
    call_sid = request.POST.get("CallSid", "")
    logger.info("Answer call", extra={"session_id": session_id, "call_sid": call_sid})
    session = _get_session(session_id)
    if session:
        session.call_sid = call_sid
        session.status = "in-progress"
        session.save()
    return HttpResponse(
        twiml_gather("Hello! I am your AI assistant. How can I help you today?", session_id),
        content_type="text/xml",
    )


@csrf_exempt
def respond_to_speech(request):
    turn_started_at = time.perf_counter()
    try:
        session_id = request.GET.get("session_id", "")
        user_speech = request.POST.get("SpeechResult", "").strip()

        if not user_speech:
            return HttpResponse(
                twiml_hangup("I didn't hear anything. Goodbye!"),
                content_type="text/xml",
            )

        session = _get_session(session_id)
        if not session:
            return HttpResponse(
                twiml_hangup("Session error. Goodbye!"),
                content_type="text/xml",
            )

        retrieved = retrieve(user_speech, top_k=3)
        context_texts = [r[0] for r in retrieved]
        system_prompt = build_system_prompt(retrieved)

        session.full_transcript.append({"role": "user", "content": user_speech})
        messages = [{"role": "system", "content": system_prompt}] + session.full_transcript

        ai_result = generate_response(messages, model_key=session.model_key)
        response_text = ai_result.get("content", "")

        turn = TurnMetrics(
            turn_index=len(session.turns),
            user_input=user_speech,
            ai_response=response_text,
            llm_latency_ms=ai_result.get("llm_latency_ms"),
            total_latency_ms=round((time.perf_counter() - turn_started_at) * 1000, 2),
            prompt_tokens=ai_result.get("prompt_tokens"),
            completion_tokens=ai_result.get("completion_tokens"),
            total_tokens=ai_result.get("total_tokens"),
            hallucination_score=None,
            faithfulness_score=None,
            context_used=context_texts,
            evaluation_notes="Evaluation scheduled asynchronously.",
        )

        session.turns.append(turn)
        session.full_transcript.append({"role": "assistant", "content": response_text})
        session.model_id = ai_result.get("model")
        session.model_provider = ai_result.get("provider")
        session.save()

        latency_log.info(
            "session=%s turn=%d model=%s llm_ms=%.1f total_ms=%.1f tokens=%d/%d/%d",
            session_id,
            turn.turn_index,
            ai_result.get("model", "-"),
            turn.llm_latency_ms,
            turn.total_latency_ms,
            ai_result.get("prompt_tokens", 0),
            ai_result.get("completion_tokens", 0),
            ai_result.get("total_tokens", 0),
        )

        threading.Thread(
            target=_score_and_persist_turn,
            args=(session.id, turn.turn_index, response_text, context_texts, user_speech),
            daemon=True,
        ).start()

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
        return HttpResponse(
            twiml_hangup("Sorry, an error occurred."),
            content_type="text/xml",
        )


@csrf_exempt
def call_status(request):
    call_sid = request.POST.get("CallSid", "")
    status = request.POST.get("CallStatus", "")
    duration = request.POST.get("CallDuration", "0")
    try:
        s = CallSession.objects.get(call_sid=call_sid)
        s.status = status
        s.call_duration_s = float(duration)
        s.finalize()
    except Exception:
        logger.exception("Status callback update failed", extra={"call_sid": call_sid})
    return HttpResponse("OK")
