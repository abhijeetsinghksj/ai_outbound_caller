import logging
import os
import sys

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class CallsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "calls"

    _warmed = False

    def ready(self):
        if CallsConfig._warmed:
            return

        # Avoid duplicate warmup when Django autoreloader parent process is active.
        if "runserver" in sys.argv and os.environ.get("RUN_MAIN") != "true":
            return

        CallsConfig._warmed = True
        try:
            from evaluation.eval_service import score_response

            print("Preloading evaluation model...")
            score_response("warmup", ["warmup"], "warmup")
            print("Evaluation model loaded.")
        except Exception:
            logger.exception("Evaluation model warmup failed")
