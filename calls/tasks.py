"""
Celery background tasks for the calls app.

score_and_persist_turn
──────────────────────
Fires immediately after respond_to_speech returns TwiML.
Does everything that used to block the hot path:
  1. NLI hallucination scoring (3–8 s)
  2. MongoDB write (200–500 ms)

The view writes only to Redis (~5 ms) and returns TwiML.
This task owns the single DB write per turn.
"""
import logging

from celery import shared_task
from mongoengine.connection import get_db
from bson import ObjectId

from calls.session_cache import get_session, save_session_to_cache, session_to_dict

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=2, default_retry_delay=5)
def score_and_persist_turn(
    self,
    session_id: str,
    turn_index: int,
    ai_text: str,
    context_texts: list,
    user_speech: str,
) -> None:
    """
    Background: score the AI response, write final session state to MongoDB,
    and refresh Redis with the scored turn.

    Args:
        session_id:    MongoDB ObjectId string of the CallSession.
        turn_index:    Index of the turn to score (0-based).
        ai_text:       The AI response text to evaluate.
        context_texts: KB chunks used as grounding context.
        user_speech:   The caller's original utterance.
    """
    try:
        logger.info("[TASK] START score turn=%d session=%s", turn_index, session_id)

        # ── 1. Score (slow: 3–8 s, runs here so the view doesn't wait) ────
        from evaluation.eval_service import score_response
        eval_result = score_response(ai_text, context_texts, user_speech)

        hallucination = eval_result.get("hallucination_score")
        faithfulness  = eval_result.get("faithfulness_score")
        notes         = eval_result.get("notes", "")

        # ── 2. Fetch session from Redis (view already primed it) ───────────
        session = get_session(session_id)
        if session is None:
            logger.error("[TASK] Session not found in cache or DB: %s", session_id)
            return

        if turn_index >= len(session.turns):
            logger.warning(
                "[TASK] turn_index=%d out of range (len=%d) session=%s",
                turn_index, len(session.turns), session_id,
            )
            return

        # ── 3. Stamp scores onto the turn ──────────────────────────────────
        session.turns[turn_index].hallucination_score = hallucination
        session.turns[turn_index].faithfulness_score  = faithfulness
        session.turns[turn_index].evaluation_notes    = notes

        # ── 4. Persist to MongoDB via replace_one ──────────────────────────
        # replace_one avoids mongoengine's _created / _changed_fields tracking
        # issues when working with a reconstructed (non-DB-loaded) document.
        mongo_doc = session.to_mongo()
        get_db()["call_sessions"].replace_one(
            {"_id": ObjectId(session_id)},
            mongo_doc,
            upsert=True,   # safe guard: creates doc if somehow missing
        )

        # ── 5. Refresh Redis with the now-scored state ─────────────────────
        save_session_to_cache(session_id, session)

        logger.info(
            "[TASK] DONE  turn=%d session=%s hallucination=%.4f faithfulness=%.4f",
            turn_index, session_id,
            hallucination or 0.0,
            faithfulness  or 0.0,
        )

    except Exception as exc:
        logger.exception(
            "[TASK] FAILED turn=%d session=%s — retrying (%d/%d)",
            turn_index, session_id,
            self.request.retries, self.max_retries,
        )
        raise self.retry(exc=exc)
