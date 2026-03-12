"""
scripts/index_kb.py
-------------------
Standalone script to (re)build the ChromaDB vector index from the documents
in knowledge_base/docs/.

Run this whenever you add, update, or remove documents so the vector store
stays in sync with the latest content.

Usage
-----
# Build index only if ChromaDB is empty (safe to run on every deploy)
python scripts/index_kb.py

# Force a full rebuild (use after updating or deleting docs)
python scripts/index_kb.py --force
"""

import sys
import os
import argparse
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
        help="Delete the existing collection and rebuild from scratch.",
    )
    args = parser.parse_args()

    print(f"[index_kb] KB docs dir  : {settings.KNOWLEDGE_BASE_DIR}")
    print(f"[index_kb] ChromaDB dir : {settings.CHROMA_DB_DIR}")

    if args.force:
        print("[index_kb] --force flag set. Full rebuild in progress...")
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
    build_index(force=args.force)
    elapsed = time.perf_counter() - t0

    collection = _get_collection()
    print(
        f"[index_kb] Done. {collection.count()} chunks in ChromaDB "
        f"(took {elapsed:.2f}s)."
    )


if __name__ == "__main__":
    main()
