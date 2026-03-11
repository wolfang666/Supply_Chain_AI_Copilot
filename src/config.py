"""
config.py
─────────
Single source of truth for all application settings.
Loads values from the .env file (via python-dotenv) and exposes
typed constants used throughout the project.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent   
load_dotenv(_ROOT / ".env", override=False)       

GROQ_API_KEY: str   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RAG_TOP_K: int       = int(os.getenv("RAG_TOP_K", "6"))

DATA_DIR: Path          = _ROOT / "data"
SAMPLE_DATASET: Path    = DATA_DIR / "sample_dataset.csv"

DELAY_THRESHOLD_DAYS: int = int(os.getenv("DELAY_THRESHOLD_DAYS", "3"))
TOP_DESTINATIONS_N: int   = int(os.getenv("TOP_DESTINATIONS_N", "10"))


APP_TITLE: str  = "Supply Chain AI Copilot"
APP_ICON: str   = "🔗"
APP_LAYOUT: str = "wide"


def is_groq_configured() -> bool:
    """Return True if a non-empty GROQ_API_KEY is available."""
    return bool(GROQ_API_KEY)
