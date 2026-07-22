"""Central configuration for every lesson.

Keeping model + embeddings wiring in ONE place means the lessons stay focused on the
concept being taught (chains, tools, agents, RAG) instead of repeating boilerplate.

Usage:
    from agentlab.config import get_chat_model, get_embeddings

    llm = get_chat_model()                            # default Groq chat model
    fast = get_chat_model("llama-3.1-8b-instant")     # faster / cheaper
    emb = get_embeddings()                            # local HuggingFace embeddings
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load .env once, when this module is first imported.
# NOTE: load_dotenv() does NOT override variables already set in your system
# environment. Since your GROQ_API_KEY is already exported, the system value wins.
load_dotenv()

# Default free Groq models (both have 128K context and support tool calling).
DEFAULT_CHAT_MODEL = "llama-3.3-70b-versatile"  # best all-round, reliable tool use
FAST_CHAT_MODEL = "llama-3.1-8b-instant"        # fastest / cheapest

# Local embedding model — small, fast, runs on CPU. Groq has no embeddings API,
# so RAG uses this instead of a hosted embeddings endpoint.
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _require_key(name: str) -> None:
    """Fail early with a friendly message if a required key is missing."""
    value = os.getenv(name)
    if not value or value.startswith("your_"):
        raise RuntimeError(
            f"Environment variable {name} is not set (or still a placeholder).\n"
            f"Add it to your .env file or system environment. See README.md."
        )


def get_chat_model(model: str = DEFAULT_CHAT_MODEL, temperature: float = 0.0, **kwargs):
    """Return a configured Groq chat model.

    Args:
        model: Groq model id. Try FAST_CHAT_MODEL for quick iteration.
        temperature: 0.0 = deterministic; raise it for more creative output.
        **kwargs: forwarded to ChatGroq (e.g. max_tokens, timeout, max_retries).
    """
    from langchain_groq import ChatGroq

    _require_key("GROQ_API_KEY")
    return ChatGroq(model=model, temperature=temperature, **kwargs)


def get_embeddings(model: str = DEFAULT_EMBED_MODEL):
    """Return local HuggingFace embeddings (no API key, runs on CPU).

    The first call downloads the model weights (~90 MB) and caches them.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model)
