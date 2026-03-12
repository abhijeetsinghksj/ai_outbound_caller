"""
knowledge_base/kb_service.py
-----------------------------
ChromaDB-backed knowledge-base service for the AI voice agent.

Lifecycle
---------
1.  On server start Django calls KnowledgeBaseConfig.ready() (apps.py).
2.  ready() calls init_kb() which creates ONE KnowledgeBase instance (_KB).
3.  Every voice turn calls retrieve(query) through the module-level wrapper.
    This hits the pre-built HNSW index only — zero file I/O, zero re-indexing.
4.  If chroma_db/ is empty the server prints a clear error and starts anyway,
    but retrieve() will raise RuntimeError until the index is built.

Indexing (run once, before first call)
---------------------------------------
    python scripts/index_kb.py           # first-time build
    python scripts/index_kb.py --force   # rebuild after doc changes
"""

from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
from django.conf import settings


# ---------------------------------------------------------------------------
# Module-level singleton — set ONCE by init_kb(), never recreated per request.
# ---------------------------------------------------------------------------
_KB: Optional["KnowledgeBase"] = None


# ---------------------------------------------------------------------------
# KnowledgeBase class
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """
    Wraps a persistent ChromaDB collection and a sentence-transformer encoder.

    Instantiated ONCE at Django app startup via init_kb() / AppConfig.ready().
    All retrieve() calls hit the pre-built HNSW cosine index — no file I/O
    per request, no re-indexing, minimal latency.
    """

    COLLECTION_NAME = "knowledge_base"
    ENCODER_MODEL   = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        import chromadb
        from sentence_transformers import SentenceTransformer

        chroma_dir = settings.CHROMA_DB_DIR
        os.makedirs(chroma_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(path=chroma_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        count = self._collection.count()
        if count == 0:
            raise RuntimeError(
                "\n"
                "  ╔══════════════════════════════════════════════════════╗\n"
                "  ║  [KB] ChromaDB is EMPTY — no embeddings indexed yet. ║\n"
                "  ║                                                        ║\n"
                "  ║  Run the indexing script FIRST, then restart:          ║\n"
                "  ║                                                        ║\n"
                "  ║      python scripts/index_kb.py                        ║\n"
                "  ║                                                        ║\n"
                "  ║  After indexing, restart the Django server.            ║\n"
                "  ╚══════════════════════════════════════════════════════╝\n"
            )

        print(
            f"[KB] ✓ ChromaDB loaded — collection='{self.COLLECTION_NAME}' "
            f"chunks={count}  path={chroma_dir}"
        )

        # Load encoder AFTER confirming ChromaDB has data (avoids wasting
        # ~1-2 s on model load when the user hasn't indexed yet).
        self._encoder = SentenceTransformer(self.ENCODER_MODEL)
        print(f"[KB] ✓ Encoder ready — model='{self.ENCODER_MODEL}'")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """
        Embed *query* and return the top-*top_k* most relevant KB chunks.

        Returns
        -------
        List of ``(text, source, similarity_score)`` tuples ordered by
        descending similarity.  Identical signature to the old in-memory
        implementation so ``views.py`` needs no changes.
        """
        q_emb = self._encoder.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )[0].tolist()

        n = min(top_k, self._collection.count())
        results = self._collection.query(
            query_embeddings=[q_emb],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        docs  = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        # ChromaDB cosine distance = 1 − cosine_similarity for L2-normalised
        # vectors.  all-MiniLM-L6-v2 + normalize_embeddings=True guarantees
        # this, so: similarity = 1 − distance  ∈ [0, 1], higher is better.
        return [
            (doc, meta["source"], max(0.0, 1.0 - dist))
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    @staticmethod
    def build_system_prompt(chunks: List[Tuple[str, str, float]]) -> str:
        """
        Build the system prompt prepended to every Groq LLM request.

        The prompt grounds the LLM strictly in the ChromaDB-retrieved chunks
        and is tuned for low-latency, real-time voice calls (short replies,
        spoken English, no markdown).
        """
        # --- 1. Role & strict grounding rules ---
        p = (
            "You are a professional AI voice assistant making an outbound customer call.\n"
            "Your goal is to help the caller by answering their questions clearly and accurately.\n\n"

            "STRICT RULES — follow these at all times:\n"
            "1. Answer ONLY using the information in the KNOWLEDGE BASE section below.\n"
            "2. If the answer is not in the knowledge base, say exactly: "
            "\"I don't have that information right now. "
            "I can arrange for someone to follow up with you.\"\n"
            "3. Never guess, invent, or add information not present in the knowledge base.\n"
            "4. Keep every reply SHORT — 1 to 2 sentences maximum — this is a voice call.\n"
            "5. Use plain spoken English. No bullet points, markdown, or technical jargon.\n"
            "6. Be polite, warm, and professional at all times.\n"
            "7. If the caller asks something outside the knowledge base, offer once to escalate "
            "and move on — do not apologise repeatedly.\n\n"
        )

        # --- 2. ChromaDB-retrieved context (injected dynamically each turn) ---
        if chunks:
            p += (
                "=== KNOWLEDGE BASE (retrieved via ChromaDB vector search) ===\n"
                "The following passages were selected as most relevant to the caller's question.\n"
                "Use ONLY these passages to form your answer.\n\n"
            )
            for i, (text, src, score) in enumerate(chunks, 1):
                p += f"[Chunk {i} | source: {src} | relevance: {score:.2f}]\n{text}\n\n"
            p += "=== END OF KNOWLEDGE BASE ===\n\n"
        else:
            p += (
                "=== KNOWLEDGE BASE ===\n"
                "[No relevant knowledge base content was found for this query.]\n"
                "=== END OF KNOWLEDGE BASE ===\n\n"
                "Since no knowledge base content is available, tell the caller you will arrange "
                "a follow-up. Do NOT attempt to answer from memory.\n\n"
            )

        # --- 3. Final anchor (last thing LLM reads before generating a reply) ---
        p += (
            "Remember: answer in 1-2 spoken sentences using only the knowledge base above. "
            "Do not fabricate any information.\n"
        )
        return p

    @property
    def chunk_count(self) -> int:
        """Total chunks currently stored in ChromaDB."""
        return self._collection.count()


# ---------------------------------------------------------------------------
# Singleton lifecycle — called from AppConfig.ready(), NOT per request
# ---------------------------------------------------------------------------

def init_kb() -> None:
    """
    Initialise the module-level singleton KnowledgeBase.

    Called once from KnowledgeBaseConfig.ready() when Django starts.
    Safe to call multiple times (no-op if already initialised).
    """
    global _KB
    if _KB is not None:
        return  # already initialised (e.g. dev auto-reload)

    try:
        _KB = KnowledgeBase()
    except RuntimeError as exc:
        # Print the clear "run index_kb.py" message but do NOT crash the server.
        # retrieve() will raise a helpful RuntimeError if called before indexing.
        print(str(exc))
        _KB = None


def get_kb() -> "KnowledgeBase":
    """
    Return the initialised singleton.

    Raises RuntimeError with clear instructions if init_kb() has not been
    called or the index has not been built yet.
    """
    if _KB is None:
        raise RuntimeError(
            "[KB] Knowledge base is not initialised.\n"
            "Run:  python scripts/index_kb.py\n"
            "Then restart the Django server."
        )
    return _KB


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# Keeps calls/views.py imports identical: retrieve(), build_system_prompt()
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
    """Query ChromaDB and return top-k relevant chunks. No indexing ever happens here."""
    return get_kb().retrieve(query, top_k)


def build_system_prompt(chunks: List[Tuple[str, str, float]]) -> str:
    """Build the grounded LLM system prompt from ChromaDB-retrieved chunks."""
    return KnowledgeBase.build_system_prompt(chunks)


# ---------------------------------------------------------------------------
# Index builder — called ONLY by scripts/index_kb.py, NEVER per request
# ---------------------------------------------------------------------------

CHUNK_SIZE    = 500   # words per chunk
CHUNK_OVERLAP = 50    # word overlap between consecutive chunks
UPSERT_BATCH  = 100   # ChromaDB upsert batch size


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split *text* into overlapping word-level chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += size - overlap
    return [c for c in chunks if c.strip()]


def build_index(force: bool = False) -> dict:
    """
    Read docs → chunk → embed → upsert into ChromaDB.

    Parameters
    ----------
    force : bool
        Delete and fully rebuild the collection (use after editing docs).

    Returns
    -------
    dict with keys: files (int), chunks (int), skipped (bool)

    Called exclusively by ``scripts/index_kb.py``.  Never called at
    request-time — that is the core fix for the per-request re-indexing bug.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    chroma_dir = settings.CHROMA_DB_DIR
    os.makedirs(chroma_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=chroma_dir)

    # Drop and recreate if force-rebuilding
    if force:
        try:
            client.delete_collection(KnowledgeBase.COLLECTION_NAME)
            print("[index] Existing collection deleted — rebuilding from scratch.")
        except Exception:
            pass  # collection didn't exist yet — that's fine

    collection = client.get_or_create_collection(
        name=KnowledgeBase.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if not force and collection.count() > 0:
        print(
            f"[index] ChromaDB already has {collection.count()} chunks. "
            "Pass --force to rebuild."
        )
        return {"files": 0, "chunks": collection.count(), "skipped": True}

    # Discover source documents
    kb_dir = settings.KNOWLEDGE_BASE_DIR
    files  = (
        glob.glob(os.path.join(kb_dir, "**/*.txt"), recursive=True)
        + glob.glob(os.path.join(kb_dir, "**/*.md"),  recursive=True)
    )

    if not files:
        print(f"[index] ✗ No .txt or .md files found in: {kb_dir}")
        return {"files": 0, "chunks": 0, "skipped": False}

    print(f"[index] Found {len(files)} document(s) in {kb_dir}")
    print(f"[index] Loading encoder '{KnowledgeBase.ENCODER_MODEL}' ...")
    encoder = SentenceTransformer(KnowledgeBase.ENCODER_MODEL)

    all_ids, all_docs, all_embs, all_meta = [], [], [], []
    file_count = 0

    for fpath in sorted(files):
        fname  = Path(fpath).name
        source = Path(fpath).stem
        text   = Path(fpath).read_text(encoding="utf-8", errors="ignore")
        chunks = _chunk_text(text)

        if not chunks:
            print(f"  [skip] {fname} — no content after chunking")
            continue

        embs = encoder.encode(
            chunks,
            show_progress_bar=False,
            batch_size=32,
            normalize_embeddings=True,
        )

        for idx, (chunk, emb) in enumerate(zip(chunks, embs)):
            all_ids.append(f"{source}_{idx}")
            all_docs.append(chunk)
            all_embs.append(emb.tolist())
            all_meta.append({"source": source, "chunk_index": idx})

        print(f"  [ok]   {fname} → {len(chunks)} chunk(s)")
        file_count += 1

    if not all_ids:
        print("[index] ✗ No chunks produced — check that docs contain text.")
        return {"files": file_count, "chunks": 0, "skipped": False}

    # Upsert in batches to stay memory-efficient for large KBs
    print(f"[index] Upserting {len(all_ids)} chunks into ChromaDB ...")
    for start in range(0, len(all_ids), UPSERT_BATCH):
        end = start + UPSERT_BATCH
        collection.upsert(
            ids=all_ids[start:end],
            documents=all_docs[start:end],
            embeddings=all_embs[start:end],
            metadatas=all_meta[start:end],
        )

    total = collection.count()
    print(f"[index] ✓ Index built successfully. {total} chunks from {file_count} file(s).")
    return {"files": file_count, "chunks": total, "skipped": False}
