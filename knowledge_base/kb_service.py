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
# Module-level singletons – initialised lazily on first use, then cached.
# Python guarantees a module is imported once per process, so these are
# effectively process-scoped singletons.
# ---------------------------------------------------------------------------
_encoder = None
_chroma_client = None
_collection = None
_kb_initialized = False  # prints "[KB] ChromaDB loaded" exactly once per process
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

def _chunk_text(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    """Split *text* into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks
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

def build_index(force: bool = False) -> Tuple[int, int]:
    """
    Initialise the module-level singleton KnowledgeBase.

    Parameters
    ----------
    force : bool
        Delete and rebuild the entire collection from scratch.

    Returns
    -------
    (total_chunks, total_files) tuple.
    """
    global _KB
    if _KB is not None:
        return  # already initialised (e.g. dev auto-reload)

    # Skip if data already present and not forcing a rebuild.
    if not force and collection.count() > 0:
        print(f"[KB] ChromaDB already has {collection.count()} chunks. Skipping rebuild.")
        return collection.count(), 0


def get_kb() -> "KnowledgeBase":
    """
    Return the initialised singleton.

    if not files:
        print(f"[KB] No docs found in {kb_dir}. Add .txt or .md files.")
        return 0, 0

    # When forcing, drop the old collection so stale chunks are removed.
    if force and collection.count() > 0:
        global _collection, _chroma_client
        _chroma_client.delete_collection("knowledge_base")
        _collection = _chroma_client.create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )
    return _KB


    for fpath in files:
        fname = Path(fpath).name
        source = Path(fpath).stem
        print(f"[KB] Loading: {fname}")
        text = open(fpath, encoding="utf-8", errors="ignore").read()
        raw_chunks = [c for c in _chunk_text(text) if c.strip()]
        if not raw_chunks:
            continue

        print(f"[KB] Created {len(raw_chunks)} chunks from {fname}")

        # Encode all chunks for this file in one batched call.
        embs = enc.encode(raw_chunks, show_progress_bar=False, batch_size=32)
# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# Keeps calls/views.py imports identical: retrieve(), build_system_prompt()
# ---------------------------------------------------------------------------

        file_chunk_count = 0
        for idx, (chunk, emb) in enumerate(zip(raw_chunks, embs)):
            ids.append(f"{source}_{idx}")
            documents.append(chunk)
            embeddings.append(emb.tolist())
            metadatas.append({"source": source, "chunk_index": idx})
            file_chunk_count += 1

        print(f"[KB] Created {file_chunk_count} chunks from {fname}")

    if not ids:
        print("[KB] No non-empty chunks produced from docs.")
        return 0, 0

def build_system_prompt(chunks: List[Tuple[str, str, float]]) -> str:
    """Build the grounded LLM system prompt from ChromaDB-retrieved chunks."""
    return KnowledgeBase.build_system_prompt(chunks)


# ---------------------------------------------------------------------------
# Index builder — called ONLY by scripts/index_kb.py, NEVER per request
# ---------------------------------------------------------------------------

    print(f"[KB] Total chunks indexed: {len(ids)}")
    return len(ids), len(files)


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

    RAISES RuntimeError if ChromaDB is empty — run scripts/index_kb.py first.

    Returns
    -------
    List of (text, source, similarity_score) tuples, sorted by descending
    similarity.
    """
    global _kb_initialized
    collection = _get_collection()

    if collection.count() == 0:
        raise RuntimeError(
            "[KB] FATAL: ChromaDB is empty or missing. "
            "Run this command first: python scripts/index_kb.py"
        )

    chroma_dir = settings.CHROMA_DB_DIR
    os.makedirs(chroma_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=chroma_dir)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # ChromaDB cosine distance = 1 - cosine_similarity for normalised vectors.
    # all-MiniLM-L6-v2 outputs are L2-normalised, so distance ∈ [0, 1].
    # Convert to similarity score so higher == better.
    chunks = [
        (doc, meta["source"], max(0.0, 1.0 - dist))
        for doc, meta, dist in zip(docs, metas, dists)
    ]
    return chunks

    collection = client.get_or_create_collection(
        name=KnowledgeBase.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

def build_system_prompt(chunks: List[Tuple[str, str, float]]) -> str:
    """
    Build the system prompt that is prepended to every Groq LLM request.
    """
    p = (
        "You are a helpful AI voice assistant on an outbound call. "
        "Be concise and friendly. Keep answers under 2 sentences unless asked for detail. "
        "Use ONLY the knowledge below. If unsure, say so — never fabricate.\n\n"
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
