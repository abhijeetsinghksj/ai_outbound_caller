"""
knowledge_base/apps.py
-----------------------
Django AppConfig for the knowledge_base app.

AppConfig.ready() is called ONCE when the Django server starts (after all
models and apps are loaded).  We use it to initialise the ChromaDB singleton
so that:

  - The encoder and collection are loaded into memory once per process.
  - Every subsequent request just queries the pre-warmed index.
  - No indexing, no file I/O, no embedding generation happens at request time.
"""

from django.apps import AppConfig


class KnowledgeBaseConfig(AppConfig):
    name = "knowledge_base"
    verbose_name = "Knowledge Base"

    def ready(self) -> None:
        """
        Initialise the KB singleton when Django finishes starting.

        This is the ONLY place init_kb() should be called.
        It will:
          1. Connect to the persistent ChromaDB collection.
          2. Load the sentence-transformer encoder into memory.
          3. Print a clear error (without crashing) if the index hasn't been
             built yet and remind the developer to run index_kb.py.
        """
        # Guard against double-initialisation in Django's dev auto-reloader
        # (the reloader imports apps twice; ready() is called once per process).
        from knowledge_base.kb_service import init_kb
        init_kb()
