from __future__ import annotations
import re
from typing import List

_nli_model = None

def _get_nli():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small", max_length=512)
    return _nli_model

def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def score_response(ai_response, context_chunks, user_query=""):
    if not context_chunks:
        return {"hallucination_score": None, "faithfulness_score": None, "notes": "No KB context."}
    nli     = _get_nli()
    context = " ".join(context_chunks)
    preds   = nli.predict([(context, ai_response)], apply_softmax=True)
    entail  = float(preds[0][1])
    contra  = float(preds[0][0])
    sents   = _sentences(ai_response)
    if not sents:
        supported, faith = 0, 1.0
    else:
        supported = 0
        for sent in sents:
            pairs = [(chunk, sent) for chunk in context_chunks]
            sp    = nli.predict(pairs, apply_softmax=True)
            if max(float(p[1]) for p in sp) > 0.5:
                supported += 1
        faith = round(supported / len(sents), 4)
    return {
        "hallucination_score": round(contra, 4),
        "faithfulness_score":  faith,
        "notes": f"Entailment={entail:.2f} | Contradiction={contra:.2f} | Supported={supported}/{len(sents)}",
    }
