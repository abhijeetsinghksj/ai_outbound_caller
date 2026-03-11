import time, logging, traceback
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from calls.models import CallSession, TurnMetrics
from calls.twilio_service import twiml_gather, twiml_hangup
from calls.ai_service import generate_response
from knowledge_base.kb_service import retrieve, build_system_prompt
from evaluation.eval_service import score_response

logger = logging.getLogger(__name__)

def _get_session(sid):
    try:
        return CallSession.objects.get(id=sid)
    except Exception as e:
        print(f"[SESSION ERROR] {e}")
        return None

@csrf_exempt
def answer_call(request):
    session_id = request.GET.get("session_id","")
    call_sid   = request.POST.get("CallSid","")
    print(f"[ANSWER] session_id={session_id} call_sid={call_sid}")
    session = _get_session(session_id)
    if session:
        session.call_sid = call_sid
        session.status   = "in-progress"
        session.save()
    return HttpResponse(
        twiml_gather("Hello! I am your A I assistant. How can I help you today?", session_id),
        content_type="text/xml")

@csrf_exempt
def respond_to_speech(request):
    print("[RESPOND] enter respond_to_speech")
    try:
        print(f"[RESPOND] request.method={request.method}")
        print(f"[RESPOND] GET params={request.GET.dict()}")
        print(f"[RESPOND] POST params keys={list(request.POST.keys())}")

        session_id = request.GET.get("session_id", "")
        print(f"[RESPOND] parsed session_id={session_id}")

        user_speech = request.POST.get("SpeechResult", "")
        print(f"[RESPOND] raw SpeechResult='{user_speech}'")
        user_speech = user_speech.strip()
        print(f"[RESPOND] cleaned SpeechResult='{user_speech}'")

        if not user_speech:
            print("[RESPOND] no user_speech, hanging up")
            return HttpResponse(
                twiml_hangup("I didn't hear anything. Goodbye!"),
                content_type="text/xml",
            )

        print("[RESPOND] fetching session from DB")
        session = _get_session(session_id)
        if not session:
            print("[RESPOND] session not found, hanging up")
            return HttpResponse(
                twiml_hangup("Session error. Goodbye!"),
                content_type="text/xml",
            )

        print(f"[RESPOND] session loaded, model_key={session.model_key}")
        print("[RESPOND] calling knowledge_base.retrieve")
        retrieved = retrieve(user_speech, top_k=3)
        print(f"[RESPOND] retrieve returned {len(retrieved)} items")

        context_texts = [r[0] for r in retrieved]
        print(f"[RESPOND] built context_texts, count={len(context_texts)}")

        system_prompt = build_system_prompt(retrieved)
        print("[RESPOND] built system_prompt")

        session.full_transcript.append({"role": "user", "content": user_speech})
        messages = [{"role": "system", "content": system_prompt}] + session.full_transcript
        print(f"[RESPOND] messages length={len(messages)}")

        print("[RESPOND] calling generate_response")
        ai_result = generate_response(messages, model_key=session.model_key)
        print(f"[RESPOND] generate_response returned keys={list(ai_result.keys())}")

        ai_text = ai_result.get("content", "")
        print(f"[RESPOND] AI content preview='{ai_text[:120]}'")

        print("[RESPOND] calling score_response")
        eval_result = score_response(ai_text, context_texts, user_speech)
        print(f"[RESPOND] score_response result keys={list(eval_result.keys())}")

        print("[RESPOND] building TurnMetrics")
        turn = TurnMetrics(
            turn_index=len(session.turns),
            user_input=user_speech,
            ai_response=ai_text,
            llm_latency_ms=ai_result.get("llm_latency_ms"),
            total_latency_ms=round((time.perf_counter()) * 1000, 2),
            prompt_tokens=ai_result.get("prompt_tokens"),
            completion_tokens=ai_result.get("completion_tokens"),
            total_tokens=ai_result.get("total_tokens"),
            hallucination_score=eval_result.get("hallucination_score"),
            faithfulness_score=eval_result.get("faithfulness_score"),
            context_used=context_texts,
            evaluation_notes=eval_result.get("notes"),
        )

        print("[RESPOND] appending turn to session.turns")
        session.turns.append(turn)
        print("[RESPOND] appending assistant message to full_transcript")
        session.full_transcript.append({"role": "assistant", "content": ai_text})

        print("[RESPOND] saving session")
        session.save()

        end_words = ["bye", "goodbye", "hang up", "that's all", "nothing else", "thank you bye"]
        print(f"[RESPOND] checking end_words against user_speech='{user_speech.lower()}'")
        if any(w in user_speech.lower() for w in end_words):
            print("[RESPOND] detected end_words, hanging up")
            return HttpResponse(
                twiml_hangup(ai_text + " Goodbye!"),
                content_type="text/xml",
            )

        print("[RESPOND] continuing conversation with twiml_gather")
        return HttpResponse(twiml_gather(ai_text, session_id), content_type="text/xml")

    except Exception:
        print("[RESPOND ERROR] exception in respond_to_speech")
        print(traceback.format_exc())
        return HttpResponse(
            twiml_hangup("Sorry, an error occurred."),
            content_type="text/xml",
        )

@csrf_exempt
def call_status(request):
    call_sid = request.POST.get("CallSid","")
    status   = request.POST.get("CallStatus","")
    duration = request.POST.get("CallDuration","0")
    try:
        s = CallSession.objects.get(call_sid=call_sid)
        s.status = status
        s.call_duration_s = float(duration)
        s.finalize()
    except Exception as e:
        print(f"[STATUS ERROR] {e}")
    return HttpResponse("OK")
