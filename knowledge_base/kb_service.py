import os
import glob
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
from django.conf import settings

# ---------------------------------------------------------------------------
# Module-level singletons – initialised lazily to keep import time near zero,
# which is critical for real-time voice latency.
# ---------------------------------------------------------------------------
_encoder = None
_chroma_client = None
_collection = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_encoder():
    """Lazy-load the sentence-transformer model (cached after first load)."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def _get_collection():
    """
    Return (and lazily create) the persistent ChromaDB collection.

    Uses cosine space so that distances returned by .query() are
    1 - cosine_similarity, meaning lower distance == more similar.
    """
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    import chromadb

    chroma_dir = settings.CHROMA_DB_DIR
    os.makedirs(chroma_dir, exist_ok=True)

    _chroma_client = chromadb.PersistentClient(path=chroma_dir)
    _collection = _chroma_client.get_or_create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def _chunk_text(text: str, size: int = 400, overlap: int = 50) -> List[str]:
    """Split *text* into overlapping word-level chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(force: bool = False) -> None:
    """
    Load all .txt / .md files from KNOWLEDGE_BASE_DIR, chunk them, embed each
    chunk with sentence-transformers, and upsert into ChromaDB.

    The collection persists on disk, so subsequent server restarts skip
    re-indexing unless *force=True* is passed (e.g. when docs change).

    Parameters
    ----------
    force : bool
        Delete and rebuild the entire collection from scratch.
    """
    collection = _get_collection()

    # Skip if data already present and not forcing a rebuild.
    if not force and collection.count() > 0:
        print(f"[KB] ChromaDB already has {collection.count()} chunks. Skipping rebuild.")
        return

    kb_dir = settings.KNOWLEDGE_BASE_DIR
    os.makedirs(kb_dir, exist_ok=True)

    files = (
        glob.glob(os.path.join(kb_dir, "**/*.txt"), recursive=True)
        + glob.glob(os.path.join(kb_dir, "**/*.md"), recursive=True)
    )

    if not files:
        print(f"[KB] No docs found in {kb_dir}. Add .txt or .md files.")
        return

    # When forcing, drop the old collection so stale chunks are removed.
    if force and collection.count() > 0:
        global _collection, _chroma_client
        _chroma_client.delete_collection("knowledge_base")
        _collection = _chroma_client.create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )
        collection = _collection
        print("[KB] Cleared existing ChromaDB collection for rebuild.")

    enc = _get_encoder()
    ids: List[str] = []
    documents: List[str] = []
    embeddings: List[List[float]] = []
    metadatas: List[dict] = []

    for fpath in files:
        source = Path(fpath).stem
        text = open(fpath, encoding="utf-8", errors="ignore").read()
        raw_chunks = [c for c in _chunk_text(text) if c.strip()]
        if not raw_chunks:
            continue

        # Encode all chunks for this file in one batched call.
        embs = enc.encode(raw_chunks, show_progress_bar=False, batch_size=32)

        for idx, (chunk, emb) in enumerate(zip(raw_chunks, embs)):
            ids.append(f"{source}_{idx}")
            documents.append(chunk)
            embeddings.append(emb.tolist())
            metadatas.append({"source": source, "chunk_index": idx})

    if not ids:
        print("[KB] No non-empty chunks produced from docs.")
        return

    # Upsert in batches of 100 to stay memory-efficient for large KBs.
    BATCH = 100
    for start in range(0, len(ids), BATCH):
        end = start + BATCH
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"[KB] Indexed {len(ids)} chunks from {len(files)} file(s) into ChromaDB.")


def retrieve(query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
    """
    Embed *query* and return the top-*k* most relevant KB chunks.

    Triggers a one-time index build on first call if ChromaDB is empty.

    Returns
    -------
    List of (text, source, similarity_score) tuples, sorted by descending
    similarity.  Signature is backward-compatible with the previous
    in-memory implementation so callers (views.py) need no changes.
    """
    collection = _get_collection()

    # Auto-build on first request (covers cold-start after deploy).
    if collection.count() == 0:
        build_index()

    if collection.count() == 0:
        return []

    enc = _get_encoder()
    q_emb = enc.encode([query], show_progress_bar=False)[0].tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # ChromaDB cosine distance = 1 - cosine_similarity for normalised vectors.
    # all-MiniLM-L6-v2 outputs are L2-normalised, so distance ∈ [0, 1].
    # Convert to similarity score so higher == better (matches old API).
    chunks = [
        (doc, meta["source"], max(0.0, 1.0 - dist))
        for doc, meta, dist in zip(docs, metas, dists)
    ]
    return chunks


def build_system_prompt(chunks: List[Tuple[str, str, float]]) -> str:
    """
    Build the system prompt that is prepended to every Groq LLM request.

    Accepts the same tuple format returned by retrieve() so this function
    requires no changes when switching from in-memory to ChromaDB retrieval.
    """
    p = (
        "You are a helpful AI voice assistant on an outbound call. "
        "Be concise and friendly. Keep answers under 2 sentences unless asked for detail. "
        "Use ONLY the knowledge below. If unsure, say so — never fabricate.\n\n"
    )
    if chunks:
        p += "=== KNOWLEDGE BASE ===\n"
        for i, (text, src, _) in enumerate(chunks, 1):
            p += f"[{i}] (source: {src})\n{text}\n\n"
        p += "=== END ===\n"
    else:
        p += "[No KB docs loaded.]\n"
    return p
