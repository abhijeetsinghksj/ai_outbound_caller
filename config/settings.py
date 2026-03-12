import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "fallback-secret")
DEBUG = True
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "calls",
    "knowledge_base",
    "evaluation",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [],
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": ["django.template.context_processors.request"]},
}]

import mongoengine
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "ai_outbound_calls")
mongoengine.connect(db=MONGO_DB_NAME, host=MONGO_URI, alias="default")

DATABASES = {"default": {"ENGINE": "django.db.backends.dummy"}}

TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
PUBLIC_BASE_URL     = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
ACTIVE_MODEL   = os.getenv("ACTIVE_MODEL", "groq_llama")

MODEL_CONFIGS = {
    "groq_llama": {
        "provider":     "groq",
        "model_id":     "llama-3.1-8b-instant",
        "display_name": "Llama 3.1 8B (Groq - Fast)",
    },
    "groq_llama70": {
        "provider":     "groq",
        "model_id":     "llama-3.3-70b-versatile",
        "display_name": "Llama 3.3 70B (Groq - Smart)",
    },
}

KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", str(BASE_DIR / "knowledge_base" / "docs"))
CHROMA_DB_DIR      = os.getenv("CHROMA_DB_DIR",      str(BASE_DIR / "knowledge_base" / "chroma_db"))

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

