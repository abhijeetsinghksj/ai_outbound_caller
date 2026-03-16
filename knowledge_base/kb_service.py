"""ChromaDB-backed runtime retrieval service for the knowledge base."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from django.conf import settings


class KnowledgeBase:
    """Runtime KB client: load encoder + Chroma collection, then retrieve chunks."""

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
        self._encoder = SentenceTransformer(self.ENCODER_MODEL)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Return top matching chunks as (text, source, similarity_score)."""
        if not query or not query.strip():
            return []

        total_chunks = self._collection.count()
        if total_chunks == 0:
            return []

        query_embedding = self._encoder.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )[0].tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total_chunks),
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        return [
            (doc, (meta or {}).get("source", "unknown"), max(0.0, 1.0 - float(dist)))
            for doc, meta, dist in zip(docs, metas, dists)
        ]


_KB: Optional[KnowledgeBase] = None


def init_kb() -> None:
    """Initialize singleton KB once per process."""
    global _KB
    if _KB is None:
        _KB = KnowledgeBase()


def get_kb() -> KnowledgeBase:
    """Return initialized singleton KB instance."""
    if _KB is None:
        raise RuntimeError("KB not initialized")
    return _KB


def retrieve(query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
    """Module-level retrieval wrapper."""
    return get_kb().retrieve(query, top_k)


def build_system_prompt(chunks: List[Tuple[str, str, float]]) -> str:
    """Build the LLM system prompt grounded in retrieved KB chunks."""
    prompt = (
        "You are a professional AI voice assistant making an outbound customer call.\n"
        "Answer briefly, politely, and only using the provided knowledge base context.\n"
        "If the answer is not present, say: \"I don't have that information right now. "
        "I can arrange for someone to follow up with you.\"\n\n"
    )

    if chunks:
        prompt += "=== KNOWLEDGE BASE ===\n"
        for i, (text, src, score) in enumerate(chunks, 1):
            prompt += f"[{i}] source={src} relevance={score:.2f}\n{text}\n\n"
        prompt += "=== END KNOWLEDGE BASE ===\n"
    else:
        prompt += (
            "=== KNOWLEDGE BASE ===\n"
            "[No relevant chunks found.]\n"
            "=== END KNOWLEDGE BASE ===\n"
        )

    prompt += "Keep your response to 1-2 spoken sentences."
    return prompt
