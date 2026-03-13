"""
knowledge_base/kb_service.py
-----------------------------
ChromaDB-backed knowledge-base service for the AI voice agent.

Lifecycle
---------
1.  On server start Django calls KnowledgeBaseConfig.ready() (apps.py).
2.  ready() calls init_kb() which creates ONE KnowledgeBase instance (_KB).
3.  Every voice turn calls retrieve(query) through the module-level wrapper.
    This hits the pre-built HNSW index only - zero file I/O, zero re-indexing.
4.  If chroma_db/ is empty the server prints a clear error and starts anyway,
    but retrieve() will raise RuntimeError until the index is built.

Indexing (run once, before first call)
---------------------------------------
    python scripts/index_kb.py           # first-time build
    python scripts/index_kb.py --force   # rebuild after doc changes
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
from django.conf import settings

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
UPSERT_BATCH = 100

_KB: Optional["KnowledgeBase"] = None


class KnowledgeBase:
    """
    Wraps a persistent ChromaDB collection and a sentence-transformer encoder.
    Instantiated ONCE at Django app startup via init_kb() / AppConfig.ready().
    """

    COLLECTION_NAME = "knowledge_base"
    ENCODER_MODEL = "all-MiniLM-L6-v2"

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
                "[KB] ChromaDB is EMPTY - no embeddings indexed yet.\n"
                "Run the indexing script FIRST, then restart:\n"
                "    python scripts/index_kb.py\n"
                "After indexing, restart the Django server.\n"
            )

        print(
            f"[KB] ChromaDB loaded. {count} chunks ready. Skipping rebuild."
        )

        self._encoder = SentenceTransformer(self.ENCODER_MODEL)
        print(f"[KB] Encoder ready - model='{self.ENCODER_MODEL}'")

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """
        Embed query and return the top-k most relevant KB chunks.
        Returns list of (text, source, similarity_score) tuples.
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

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        return [
            (doc, meta["source"], max(0.0, 1.0 - dist))
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    @staticmethod
    def build_system_prompt(chunks: List[Tuple[str, str, float]]) -> str:
        """
        Build the system prompt prepended to every Groq LLM request.
        """
        p = (
            "You are a professional AI voice assistant making an outbound customer call.\n"
            "Your goal is to help the caller by answering their questions clearly and accurately.\n\n"
            "STRICT RULES - follow these at all times:\n"
            "1. Answer ONLY using the information in the KNOWLEDGE BASE section below.\n"
            "2. If the answer is not in the knowledge base, say exactly: "
            "\"I don't have that information right now. "
            "I can arrange for someone to follow up with you.\"\n"
            "3. Never guess, invent, or add information not present in the knowledge base.\n"
            "4. Keep every reply SHORT - 1 to 2 sentences maximum - this is a voice call.\n"
            "5. Use plain spoken English. No bullet points, markdown, or technical jargon.\n"
            "6. Be polite, warm, and professional at all times.\n\n"
        )

        if chunks:
            p += (
                "=== KNOWLEDGE BASE (retrieved via ChromaDB vector search) ===\n"
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
# Singleton lifecycle - called from AppConfig.ready(), NOT per request
# ---------------------------------------------------------------------------

def init_kb() -> None:
    """
    Initialise the module-level singleton KnowledgeBase.
    Called once from KnowledgeBaseConfig.ready().
    """
    global _KB
    if _KB is not None:
        return
    try:
        _KB = KnowledgeBase()
    except RuntimeError as e:
        print(f"[KB] WARNING: Could not connect to ChromaDB on startup: {e}")


def get_kb() -> "KnowledgeBase":
    """Return the initialised singleton."""
    global _KB
    if _KB is None:
        raise RuntimeError(
            "[KB] Knowledge base not initialised. "
            "Run python scripts/index_kb.py first, then restart the server."
        )
    return _KB


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# Keeps calls/views.py imports identical: retrieve(), build_system_prompt()
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
    """Retrieve top-k relevant chunks from ChromaDB for the given query."""
    return get_kb().retrieve(query, top_k)


def build_system_prompt(chunks: List[Tuple[str, str, float]]) -> str:
    """Build the grounded LLM system prompt from ChromaDB-retrieved chunks."""
    return KnowledgeBase.build_system_prompt(chunks)


# ---------------------------------------------------------------------------
# Chunk helper - used by scripts/index_kb.py ONLY, never at runtime
# ---------------------------------------------------------------------------

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c for c in chunks if c.strip()]
