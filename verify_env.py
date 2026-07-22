"""Environment verification — run this first.

    uv run python verify_env.py

Checks:
  1. Python version is 3.12.x / 3.13.x
  2. Which API keys are present (GROQ required; TAVILY / LANGSMITH optional)
  3. A live round-trip to the Groq API actually works
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"


def check_python() -> bool:
    v = sys.version_info
    ok = (v.major, v.minor) in {(3, 12), (3, 13)}
    tag = OK if ok else FAIL
    print(f"{tag} Python {v.major}.{v.minor}.{v.micro}  (want >=3.12,<3.14)")
    return ok


def key_present(name: str) -> bool:
    val = os.getenv(name)
    return bool(val) and not val.startswith("your_")


def check_keys() -> bool:
    groq = key_present("GROQ_API_KEY")
    print(f"{OK if groq else FAIL} GROQ_API_KEY {'found' if groq else 'MISSING (required)'}")

    tavily = key_present("TAVILY_API_KEY")
    print(f"{OK if tavily else WARN} TAVILY_API_KEY "
          f"{'found' if tavily else 'missing (needed only for lesson 05)'}")

    langsmith = key_present("LANGSMITH_API_KEY")
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    print(f"{OK if langsmith else WARN} LANGSMITH_API_KEY "
          f"{'found' if langsmith else 'missing (optional, tracing)'}")
    if tracing and not langsmith:
        print(f"{WARN} LANGSMITH_TRACING=true but no LANGSMITH_API_KEY — tracing will error.")
    return groq


def check_groq_roundtrip() -> bool:
    print("\nPinging Groq (llama-3.1-8b-instant)...")
    try:
        from agentlab.config import get_chat_model

        llm = get_chat_model("llama-3.1-8b-instant")
        resp = llm.invoke("Reply with exactly: pong")
        print(f"{OK} Groq says: {resp.content!r}")
        return True
    except Exception as e:  # noqa: BLE001 - we want to surface any failure clearly
        print(f"{FAIL} Groq call failed: {type(e).__name__}: {e}")
        return False


def main() -> int:
    print("=" * 60)
    print("  LangChain + Groq lab — environment check")
    print("=" * 60)
    py_ok = check_python()
    print()
    groq_key_ok = check_keys()

    if not groq_key_ok:
        print("\nCannot test Groq without GROQ_API_KEY. Fix that and re-run.")
        return 1

    groq_ok = check_groq_roundtrip()

    print("\n" + "=" * 60)
    if py_ok and groq_ok:
        print("  All good — you're ready. Try: uv run python examples/01_hello_groq.py")
        return 0
    print("  Some checks failed — see messages above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
