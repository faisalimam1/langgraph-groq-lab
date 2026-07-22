"""Lesson 05 — Your first real agent: ReAct + web search.

LangGraph's prebuilt `create_react_agent` wires up the loop from lesson 04 for you:
the model reasons, calls tools, reads results, and repeats until it can answer.
Here the tool is Tavily web search, so the agent can answer questions about current events.

Requires TAVILY_API_KEY (free at https://app.tavily.com).

Run:  uv run python examples/05_react_agent_tavily.py "your question here"
"""

import sys

from langchain.agents import create_agent
from langchain_tavily import TavilySearch

from agentlab.config import get_chat_model


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What are the newest models available on Groq right now?"

    llm = get_chat_model()

    # A single tool: web search returning up to 3 LLM-friendly results.
    search = TavilySearch(max_results=3)

    # create_agent returns a compiled LangGraph you can .invoke() / .stream().
    agent = create_agent(
        model=llm,
        tools=[search],
        system_prompt="You are a helpful research assistant. Use web search when you need current facts, then answer clearly and cite sources.",
    )

    print(f"Question: {question}\n")
    print("Agent is thinking (watch it call the search tool)...\n")

    # stream_mode="values" yields the full state after each step; we print new messages.
    seen = 0
    final_state = None
    for state in agent.stream({"messages": [("user", question)]}, stream_mode="values"):
        final_state = state
        msgs = state["messages"]
        for m in msgs[seen:]:
            role = m.type  # 'human' | 'ai' | 'tool'
            if role == "ai" and m.tool_calls:
                for tc in m.tool_calls:
                    print(f"  🔎 search: {tc['args'].get('query', tc['args'])}")
            elif role == "tool":
                preview = str(m.content)[:120].replace("\n", " ")
                print(f"  📄 results: {preview}...")
        seen = len(msgs)

    print("\n" + "=" * 60)
    print("Answer:\n")
    print(final_state["messages"][-1].content)


if __name__ == "__main__":
    main()
