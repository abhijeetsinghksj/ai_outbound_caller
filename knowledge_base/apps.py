from django.apps import AppConfig


class KnowledgeBaseConfig(AppConfig):
    name = "knowledge_base"

    def ready(self):
        """
        Pre-warm the ChromaDB collection once at Django startup.

        This runs once per process (Django calls ready() once).
        It establishes the persistent client connection and logs the chunk
        count so operators can confirm the index is healthy at boot time.
        """
        try:
            from knowledge_base.kb_service import _get_collection
            collection = _get_collection()
            count = collection.count()
            if count > 0:
                print(f"[KB] ChromaDB loaded. {count} chunks ready. Skipping rebuild.")
            else:
                print(
                    "[KB] WARNING: ChromaDB is empty. "
                    "Run: python scripts/index_kb.py"
                )
        except Exception as exc:
            print(f"[KB] WARNING: Could not connect to ChromaDB on startup: {exc}")
