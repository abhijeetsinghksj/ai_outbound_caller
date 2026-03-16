from django.apps import AppConfig


class KnowledgeBaseConfig(AppConfig):
    name = "knowledge_base"

    def ready(self):
        """
        Initialise the ChromaDB-backed knowledge base once at Django startup.
        """
        from .kb_service import init_kb

        init_kb()
