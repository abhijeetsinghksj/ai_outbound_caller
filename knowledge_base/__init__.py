# Tell Django to use KnowledgeBaseConfig as this app's AppConfig.
# This ensures AppConfig.ready() fires on server start, which initialises
# the ChromaDB singleton once — not on every request.
default_app_config = "knowledge_base.apps.KnowledgeBaseConfig"
