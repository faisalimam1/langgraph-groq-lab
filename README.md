# LangChain + LangGraph Agent Lab (Groq-powered) 🚀

[![Repo](https://img.shields.io/badge/GitHub-langgraph--groq--lab-181717?logo=github)](https://github.com/faisalimam1/langgraph-groq-lab)

A hands-on lab for learning to build **agents and agentic systems** with
[LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/),
using the **free [Groq](https://console.groq.com/) API** for fast LLM inference.

Everything here runs on free tiers:

| Concern            | What we use                                                            |
| ------------------ | ---------------------------------------------------------------------- |
| LLM inference      | **Groq** (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`)           |
| Web search tool    | **Tavily** (free tier)                                                 |
| Embeddings (RAG)   | **local HuggingFace** `all-MiniLM-L6-v2` — Groq has *no* embeddings API |
| Vector store       | **Chroma** (local, on disk)                                            |
| Tracing (optional) | **LangSmith**                                                          |

## 1. Prerequisites

- [uv](https://docs.astral.sh/uv/) (installs its own Python 3.12 — no system Python needed)
- API keys (all free):
  - `GROQ_API_KEY` — https://console.groq.com/keys
  - `TAVILY_API_KEY` — https://app.tavily.com  *(needed for the web-search agent lesson)*
  - `LANGSMITH_API_KEY` — https://smith.langchain.com  *(optional, for tracing)*

## 2. Setup

```bash
# Install dependencies (downloads Python 3.12 + all packages into .venv)
uv sync

# Create your .env from the template and fill in the keys you have
cp .env.example .env      # PowerShell: Copy-Item .env.example .env
```

> **Heads-up:** the first `uv sync` downloads PyTorch (a few hundred MB), pulled in by
> `sentence-transformers` for local embeddings. This is a one-time cost.

## 3. Verify

```bash
uv run python verify_env.py
```

This checks your Python version, which keys are present, and does a live round-trip to Groq.

## 4. Lessons

Run them in order. Each file is small and heavily commented.

| #  | File                                   | You learn                                              |
| -- | -------------------------------------- | ------------------------------------------------------ |
| 01 | `examples/01_hello_groq.py`            | Talk to a Groq model with `ChatGroq`                   |
| 02 | `examples/02_prompts_and_chains.py`    | Prompt templates + LCEL pipes (`prompt \| model \| parser`) |
| 03 | `examples/03_structured_output.py`     | Get typed Pydantic objects out of the model            |
| 04 | `examples/04_tools_and_binding.py`     | Define tools, `bind_tools()`, run a manual tool loop   |
| 05 | `examples/05_react_agent_tavily.py`    | Your first real agent: ReAct + web search              |
| 06 | `examples/06_memory_checkpointer.py`   | Short-term memory across turns (`thread_id`)           |
| 07 | `examples/07_rag_local_embeddings.py`  | RAG with local embeddings + Chroma                     |
| 08 | `examples/08_multi_agent_graph.py`     | A custom multi-node LangGraph workflow                 |

```bash
uv run python examples/01_hello_groq.py
uv run python examples/05_react_agent_tavily.py "What are the latest Groq model announcements?"
```

## 5. Notebooks (optional)

```bash
uv run jupyter lab
```

Open `notebooks/01_intro.ipynb`.

## 6. Tracing (optional)

Set `LANGSMITH_TRACING=true` in `.env` (with a valid `LANGSMITH_API_KEY`), re-run any
lesson, and watch the run appear at https://smith.langchain.com.

## Switching models

Everything goes through one helper — `src/agentlab/config.py`:

```python
from agentlab.config import get_chat_model
llm = get_chat_model()                              # default: llama-3.3-70b-versatile
fast = get_chat_model("llama-3.1-8b-instant")       # faster / cheaper
```
