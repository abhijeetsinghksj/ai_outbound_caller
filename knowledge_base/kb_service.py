import os, glob
from pathlib import Path
from typing import List, Tuple
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
from django.conf import settings

_encoder = None
_doc_chunks: List[dict] = []
_index_built = False

def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder

def _chunk_text(text, size=400, overlap=50):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks

def build_index(force=False):
    global _doc_chunks, _index_built
    if _index_built and not force:
        return
    kb_dir = settings.KNOWLEDGE_BASE_DIR
    os.makedirs(kb_dir, exist_ok=True)
    files = glob.glob(os.path.join(kb_dir,"**/*.txt"), recursive=True) + \
            glob.glob(os.path.join(kb_dir,"**/*.md"),  recursive=True)
    if not files:
        print(f"[KB] No docs in {kb_dir}. Add .txt or .md files.")
        _index_built = True
        return
    enc = _get_encoder()
    _doc_chunks = []
    for fpath in files:
        src  = Path(fpath).stem
        text = open(fpath, encoding="utf-8", errors="ignore").read()
        for chunk in _chunk_text(text):
            if chunk.strip():
                _doc_chunks.append({"text": chunk, "source": src, "embedding": None})
    texts = [c["text"] for c in _doc_chunks]
    embs  = enc.encode(texts, show_progress_bar=False, batch_size=32)
    for i, emb in enumerate(embs):
        _doc_chunks[i]["embedding"] = emb
    _index_built = True
    print(f"[KB] Indexed {len(_doc_chunks)} chunks from {len(files)} file(s).")

def retrieve(query, top_k=3):
    build_index()
    if not _doc_chunks:
        return []
    import numpy as np
    enc   = _get_encoder()
    q_emb = enc.encode([query], show_progress_bar=False)[0]
    scores = []
    for c in _doc_chunks:
        if c["embedding"] is not None:
            emb = c["embedding"]
            sim = float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb)*np.linalg.norm(emb)+1e-8))
            scores.append((c["text"], c["source"], sim))
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_k]

def build_system_prompt(chunks):
    p = ("You are a helpful AI voice assistant on an outbound call. "
         "Be concise and friendly. Keep answers under 2 sentences unless asked for detail. "
         "Use ONLY the knowledge below. If unsure, say so — never fabricate.\n\n")
    if chunks:
        p += "=== KNOWLEDGE BASE ===\n"
        for i,(text,src,_) in enumerate(chunks,1):
            p += f"[{i}] (source: {src})\n{text}\n\n"
        p += "=== END ===\n"
    else:
        p += "[No KB docs loaded.]\n"
    return p
