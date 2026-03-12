"""
scripts/index_kb.py
--------------------
Standalone script to build (or rebuild) the ChromaDB vector index from the
documents in knowledge_base/docs/.

Run this ONCE before starting the Django server for the first time, and again
whenever you add, update, or delete documents.

Usage
-----
# First-time build (or safe no-op if already built)
    python scripts/index_kb.py

# Force a full rebuild after editing / deleting docs
    python scripts/index_kb.py --force

# Verify the index without rebuilding
    python scripts/index_kb.py --verify
"""

import sys
import os
import glob
import argparse
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
        help="Delete the existing collection and rebuild from scratch.",
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

    if args.force:
        print("  Mode: FORCE REBUILD — existing index will be deleted.\n")
    else:
        print("  Mode: INCREMENTAL — skips rebuild if index already exists.\n")

    t0 = time.perf_counter()
    summary = build_index(force=args.force)
    elapsed = time.perf_counter() - t0

    collection = _get_collection()
    kb_dir = settings.KNOWLEDGE_BASE_DIR
    files = (
        glob.glob(os.path.join(kb_dir, "**/*.txt"), recursive=True)
        + glob.glob(os.path.join(kb_dir, "**/*.md"), recursive=True)
    )
    print(
        f"Index built successfully. {collection.count()} chunks from {len(files)} files. "
        f"(took {elapsed:.2f}s)"
    )
    print()
    print("=" * 60)
    if summary["skipped"]:
        print(f"  ✓ Skipped — index already has {summary['chunks']} chunks.")
        print("    Pass --force to rebuild.")
    else:
        print(f"  ✓ Index built successfully.")
        print(f"    Files   : {summary['files']}")
        print(f"    Chunks  : {summary['chunks']}")
        print(f"    Time    : {elapsed:.2f}s")
        print()
        print("  Next step: start (or restart) the Django server.")
        print("    python manage.py runserver")
    print("=" * 60)


if __name__ == "__main__":
    main()
