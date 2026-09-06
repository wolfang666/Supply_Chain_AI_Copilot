"""
config.py
─────────
Single source of truth for all application settings.

Loads values from the .env file using python-dotenv and exposes
typed constants used throughout the project.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# Project root / environment
# ============================================================

_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(
    _ROOT / ".env",
    override=False,
)


# ============================================================
# LLM — Groq
# ============================================================

GROQ_API_KEY: str = os.getenv(
    "GROQ_API_KEY",
    "",
)

GROQ_MODEL: str = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)

LLM_MAX_TOKENS: int = int(
    os.getenv(
        "LLM_MAX_TOKENS",
        "512",
    )
)

LLM_TEMPERATURE: float = float(
    os.getenv(
        "LLM_TEMPERATURE",
        "0.3",
    )
)


# ============================================================
# Embedding Model
# ============================================================

EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)


# ============================================================
# Pinecone Vector Database
# ============================================================

PINECONE_API_KEY: str = os.getenv(
    "PINECONE_API_KEY",
    "",
)

PINECONE_INDEX_NAME: str = os.getenv(
    "PINECONE_INDEX_NAME",
    "supply-chain-copilot",
)

PINECONE_CLOUD: str = os.getenv(
    "PINECONE_CLOUD",
    "aws",
)

PINECONE_REGION: str = os.getenv(
    "PINECONE_REGION",
    "us-east-1",
)

PINECONE_NAMESPACE: str = os.getenv(
    "PINECONE_NAMESPACE",
    "supply-chain",
)


# ============================================================
# Hybrid Retrieval
# ============================================================

# Number of documents retrieved from BM25.
BM25_TOP_K: int = int(
    os.getenv(
        "BM25_TOP_K",
        "20",
    )
)

# Number of documents retrieved from Pinecone.
SEMANTIC_TOP_K: int = int(
    os.getenv(
        "SEMANTIC_TOP_K",
        "20",
    )
)

# Number of final documents retained after reranking.
RERANK_TOP_K: int = int(
    os.getenv(
        "RERANK_TOP_K",
        "8",
    )
)

# Final number of chunks supplied to the generation layer.
RAG_TOP_K: int = int(
    os.getenv(
        "RAG_TOP_K",
        "6",
    )
)


# ============================================================
# Reciprocal Rank Fusion
# ============================================================

# Smoothing constant used by Reciprocal Rank Fusion:
#
#       RRF(d) = Σ 1 / (k + rank)
#
RRF_K: int = int(
    os.getenv(
        "RRF_K",
        "60",
    )
)


# ============================================================
# Cross-Encoder Reranker
# ============================================================

RERANKER_MODEL: str = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
)


# ============================================================
# Supply Chain Analytics
# ============================================================

DELAY_THRESHOLD_DAYS: int = int(
    os.getenv(
        "DELAY_THRESHOLD_DAYS",
        "3",
    )
)

TOP_DESTINATIONS_N: int = int(
    os.getenv(
        "TOP_DESTINATIONS_N",
        "10",
    )
)


# ============================================================
# Dataset
# ============================================================

DATA_DIR: Path = _ROOT / "data"

SAMPLE_DATASET: Path = (
    DATA_DIR / "sample_dataset.csv"
)


# ============================================================
# Streamlit Application
# ============================================================

APP_TITLE: str = (
    "Supply Chain AI Copilot"
)

APP_ICON: str = "🔗"

APP_LAYOUT: str = "wide"


# ============================================================
# Configuration Helpers
# ============================================================

def is_groq_configured() -> bool:
    """Return True if a non-empty GROQ API key is available."""
    return bool(
        GROQ_API_KEY.strip()
    )


def is_pinecone_configured() -> bool:
    """Return True if a non-empty Pinecone API key is available."""
    return bool(
        PINECONE_API_KEY.strip()
    )