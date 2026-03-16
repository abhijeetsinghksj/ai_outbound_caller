"""
Standalone offline script to (re)build the ChromaDB vector index from
knowledge_base/docs/.
"""

import argparse
import glob
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django

django.setup()

from django.conf import settings

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping character chunks."""
    chunks: List[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def _verify_index() -> None:
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
    patterns = ["*.md", "*.txt"]
    docs: List[str] = []
    for pattern in patterns:
        docs.extend(glob.glob(os.path.join(settings.KNOWLEDGE_BASE_DIR, pattern)))
    return [Path(p) for p in sorted(docs)]


def build_index() -> tuple[int, int]:
    import chromadb
    from sentence_transformers import SentenceTransformer

    docs = _load_documents()
    if not docs:
        raise RuntimeError(f"No documents found in {settings.KNOWLEDGE_BASE_DIR}")

    client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
    collection = client.get_or_create_collection(
        name="knowledge_base", metadata={"hnsw:space": "cosine"}
    )

    existing = collection.get(include=[])
    if existing.get("ids"):
        collection.delete(ids=existing["ids"])

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    all_ids: List[str] = []
    all_docs: List[str] = []
    all_metas: List[dict] = []

    for doc_path in docs:
        text = doc_path.read_text(encoding="utf-8", errors="ignore")
        chunks = _chunk_text(text)
        for idx, chunk in enumerate(chunks):
            all_ids.append(f"{doc_path.name}:{idx}")
            all_docs.append(chunk)
            all_metas.append({"source": doc_path.name})

    if not all_docs:
        raise RuntimeError("No non-empty chunks produced from source documents")

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
    parser.add_argument("--force", action="store_true", help="Delete existing ChromaDB and rebuild.")
    parser.add_argument("--verify", action="store_true", help="Print index stats and exit.")
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
