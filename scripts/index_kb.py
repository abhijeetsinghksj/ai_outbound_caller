"""
scripts/index_kb.py
-------------------
Standalone offline script to (re)build the ChromaDB vector index from the
documents in knowledge_base/docs/.
"""

import argparse
import glob
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Bootstrap: put the project root on sys.path so Django imports work.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django

django.setup()

from django.conf import settings
from knowledge_base.kb_service import CHUNK_OVERLAP, CHUNK_SIZE, _chunk_text


def _verify_index() -> None:
    """Print a summary of what is currently stored in ChromaDB."""
    import chromadb

    chroma_dir = settings.CHROMA_DB_DIR
    if not os.path.isdir(chroma_dir):
        print(f"[verify] ✗ chroma_db directory not found: {chroma_dir}")
        print("[verify]   Run: python scripts/index_kb.py")
        sys.exit(1)

    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        name="knowledge_base", metadata={"hnsw:space": "cosine"}
    )
    count = collection.count()

    if count == 0:
        print(f"[verify] ✗ ChromaDB exists at {chroma_dir} but contains 0 chunks.")
        print("[verify]   Run: python scripts/index_kb.py")
        sys.exit(1)

    print(f"[verify] ✓ ChromaDB has {count} chunks in collection 'knowledge_base'")
    print(f"[verify]   Path: {chroma_dir}")


def _load_documents() -> List[Path]:
    docs = sorted(glob.glob(os.path.join(settings.KNOWLEDGE_BASE_DIR, "*.md")))
    return [Path(p) for p in docs]


def build_index() -> tuple[int, int]:
    """Build ChromaDB collection from markdown docs."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    docs = _load_documents()
    if not docs:
        raise RuntimeError(f"No markdown documents found in {settings.KNOWLEDGE_BASE_DIR}")

    client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
    collection = client.get_or_create_collection(
        name="knowledge_base", metadata={"hnsw:space": "cosine"}
    )

    # Rebuild from current docs.
    existing = collection.get(include=[])
    if existing.get("ids"):
        collection.delete(ids=existing["ids"])

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    all_ids: List[str] = []
    all_docs: List[str] = []
    all_metas: List[dict] = []

    for doc_path in docs:
        text = doc_path.read_text(encoding="utf-8")
        chunks = _chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for idx, chunk in enumerate(chunks):
            all_ids.append(f"{doc_path.name}:{idx}")
            all_docs.append(chunk)
            all_metas.append({"source": doc_path.name})

    embeddings = encoder.encode(
        all_docs, show_progress_bar=False, normalize_embeddings=True
    ).tolist()

    collection.upsert(
        ids=all_ids,
        documents=all_docs,
        metadatas=all_metas,
        embeddings=embeddings,
    )

    return len(all_docs), len(docs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB knowledge-base index.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the existing ChromaDB directory and rebuild from scratch.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Print index stats and exit without rebuilding.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" ChromaDB Knowledge Base Indexer")
    print("=" * 60)
    print(f"  Docs dir   : {settings.KNOWLEDGE_BASE_DIR}")
    print(f"  ChromaDB   : {settings.CHROMA_DB_DIR}")
    print(f"  Chunk size : {CHUNK_SIZE} chars  (overlap: {CHUNK_OVERLAP})")
    print()

    if args.verify:
        _verify_index()
        return

    if args.force and os.path.exists(settings.CHROMA_DB_DIR):
        print(f"[index_kb] --force: deleting {settings.CHROMA_DB_DIR}")
        shutil.rmtree(settings.CHROMA_DB_DIR)

    os.makedirs(settings.CHROMA_DB_DIR, exist_ok=True)

    t0 = time.perf_counter()
    total_chunks, total_files = build_index()
    elapsed = time.perf_counter() - t0

    print()
    print(f"Index built successfully. {total_chunks} chunks from {total_files} files.")
    print(f"[index_kb] Done in {elapsed:.2f}s.")


if __name__ == "__main__":
    main()
