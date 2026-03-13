"""
scripts/index_kb.py
-------------------
Standalone offline script to (re)build the ChromaDB vector index from the
documents in knowledge_base/docs/.

Run this BEFORE starting the Django server, and again whenever you add,
update, or remove documents.

Usage
-----
# First-time build (or safe no-op if already built)
    python scripts/index_kb.py

# Force a full rebuild (use after updating or deleting docs)
python scripts/index_kb.py --force

# Verify current index without rebuilding
python scripts/index_kb.py --verify
"""

import sys
import os
import glob
import argparse
import shutil
import time

# ---------------------------------------------------------------------------
# Bootstrap: put the project root on sys.path so Django imports work.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

from django.conf import settings
from knowledge_base.kb_service import build_index, _chunk_text, CHUNK_SIZE, CHUNK_OVERLAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verify_index() -> None:
    """Print a summary of what is currently stored in ChromaDB."""
    import chromadb
    from knowledge_base.kb_service import KnowledgeBase

    chroma_dir = settings.CHROMA_DB_DIR
    if not os.path.isdir(chroma_dir):
        print(f"[verify] ✗ chroma_db directory not found: {chroma_dir}")
        print("[verify]   Run:  python scripts/index_kb.py")
        sys.exit(1)

    client     = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(name=KnowledgeBase.COLLECTION_NAME)
    count      = collection.count()

    if count == 0:
        print(f"[verify] ✗ ChromaDB exists at {chroma_dir} but contains 0 chunks.")
        print("[verify]   Run:  python scripts/index_kb.py")
        sys.exit(1)

    print(f"[verify] ✓ ChromaDB has {count} chunks in collection "
          f"'{KnowledgeBase.COLLECTION_NAME}'")
    print(f"[verify]   Path: {chroma_dir}")

    # Show a breakdown by source file
    results = collection.get(include=["metadatas"])
    sources: dict[str, int] = {}
    for meta in results["metadatas"]:
        src = meta.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print(f"\n  {'Source':<30} {'Chunks':>6}")
    print(f"  {'-'*30} {'------':>6}")
    for src, n in sorted(sources.items()):
        print(f"  {src:<30} {n:>6}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or rebuild the ChromaDB knowledge-base index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/index_kb.py            # build if empty\n"
            "  python scripts/index_kb.py --force    # force full rebuild\n"
            "  python scripts/index_kb.py --verify   # check index without rebuilding\n"
        ),
    )
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
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Print index stats and exit without building.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" ChromaDB Knowledge Base Indexer")
    print("=" * 60)
    print(f"  Docs dir   : {settings.KNOWLEDGE_BASE_DIR}")
    print(f"  ChromaDB   : {settings.CHROMA_DB_DIR}")
    print(f"  Chunk size : {CHUNK_SIZE} words  (overlap: {CHUNK_OVERLAP})")
    print()

    # --verify mode: just inspect and exit
    if args.verify:
        _verify_index()
        return

    # --verify: just show stats
    if args.verify:
        collection = _get_collection()
        count = collection.count()
        print(f"[index_kb] Current chunk count: {count}")
        return

    # --force: delete chroma_db dir entirely so PersistentClient starts fresh
    if args.force:
        chroma_dir = settings.CHROMA_DB_DIR
        if os.path.exists(chroma_dir):
            print(f"[index_kb] --force: deleting {chroma_dir}")
            shutil.rmtree(chroma_dir)
            # Reset the module-level singletons so _get_collection() recreates them
            import knowledge_base.kb_service as _kb_svc
            _kb_svc._chroma_client = None
            _kb_svc._collection = None
            print("[index_kb] ChromaDB directory removed. Rebuilding...")
        else:
            print("[index_kb] --force: no existing ChromaDB found, building fresh.")
    else:
        print("  Mode: INCREMENTAL — skips rebuild if index already exists.\n")

    t0 = time.perf_counter()
    total_chunks, total_files = build_index(force=False)
    elapsed = time.perf_counter() - t0

    # Confirm persistence by re-reading from disk
    collection = _get_collection()
    confirmed = collection.count()

    print()
    print(f"Index built successfully. {confirmed} chunks from {total_files} files.")
    print(f"[index_kb] Done in {elapsed:.2f}s. Confirmed {confirmed} chunks in ChromaDB.")


if __name__ == "__main__":
    main()
