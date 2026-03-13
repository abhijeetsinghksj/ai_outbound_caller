"""
scripts/index_kb.py
-------------------
Standalone offline script to (re)build the ChromaDB vector index from the
documents in knowledge_base/docs/.

Run this BEFORE starting the Django server, and again whenever you add,
update, or remove documents.

Usage
-----
# Build index only if ChromaDB is empty (safe to run on every deploy)
python scripts/index_kb.py

# Force a full rebuild (use after updating or deleting docs)
python scripts/index_kb.py --force

# Verify current index without rebuilding
python scripts/index_kb.py --verify
"""

import sys
import os
import argparse
import shutil
import time

# ---------------------------------------------------------------------------
# Bootstrap Django settings before any app imports.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

from django.conf import settings
from knowledge_base.kb_service import build_index, _get_collection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or rebuild the ChromaDB knowledge-base index."
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
    args = parser.parse_args()

    print(f"[index_kb] KB docs dir  : {settings.KNOWLEDGE_BASE_DIR}")
    print(f"[index_kb] ChromaDB dir : {settings.CHROMA_DB_DIR}")

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
        collection = _get_collection()
        existing = collection.count()
        if existing > 0:
            print(
                f"[index_kb] ChromaDB already contains {existing} chunks. "
                "Pass --force to rebuild."
            )
            return

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
