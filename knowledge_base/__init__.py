default_app_config = "knowledge_base.apps.KnowledgeBaseConfig"

# ---------------------------------------------------------------------------
# Singleton accessor — used by verification scripts and tests.
# calls/views.py imports retrieve/build_system_prompt directly from
# kb_service, which is already a module-level singleton.
# ---------------------------------------------------------------------------
_kb_instance = None


def get_kb():
    """
    Return the kb_service module as a thin singleton handle.

    The module itself is the singleton (Python imports modules once per
    process), but this function provides a stable public API that matches
    the class-based singleton pattern expected by callers.

    Usage::

        from knowledge_base import get_kb
        kb = get_kb()
        results = kb.retrieve("pricing plans")
    """
    global _kb_instance
    if _kb_instance is None:
        from knowledge_base import kb_service
        _kb_instance = kb_service
    return _kb_instance
