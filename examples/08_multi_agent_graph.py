"""Lesson 08 — A custom multi-node LangGraph workflow.

Prebuilt agents are convenient, but the real power of LangGraph is composing your own
graph of nodes with explicit control flow. Here we build a tiny 3-node pipeline:

    research  ->  write  ->  critique  -> END

Each node is just a function that takes the shared state and returns an update to it.
This "assembly line of specialists" pattern scales up to real multi-agent systems.

Run:  uv run python examples/08_multi_agent_graph.py
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agentlab.config import get_chat_model

llm = get_chat_model()


# The shared state passed between nodes. Each node reads and adds to it.
class State(TypedDict):
    topic: str
    facts: str
    draft: str
    final: str


def research_node(state: State) -> dict:
    """Specialist 1: brainstorm key facts about the topic."""
    resp = llm.invoke(f"List 3 concise bullet-point facts about: {state['topic']}")
    print("🔬 research done")
    return {"facts": resp.content}


def write_node(state: State) -> dict:
    """Specialist 2: turn the facts into a short paragraph."""
    resp = llm.invoke(
        f"Write a single engaging paragraph about '{state['topic']}' using these facts:\n{state['facts']}"
    )
    print("✍️  draft written")
    return {"draft": resp.content}


def critique_node(state: State) -> dict:
    """Specialist 3: polish the draft into a final version."""
    resp = llm.invoke(
        f"Improve this paragraph for clarity and flow. Return only the improved version:\n{state['draft']}"
    )
    print("🔎 critique + polish done")
    return {"final": resp.content}


def main() -> None:
    # Build the graph: register nodes, then wire the edges.
    graph = StateGraph(State)
    graph.add_node("research", research_node)
    graph.add_node("write", write_node)
    graph.add_node("critique", critique_node)

    graph.add_edge(START, "research")
    graph.add_edge("research", "write")
    graph.add_edge("write", "critique")
    graph.add_edge("critique", END)

    app = graph.compile()

    print("Running the research -> write -> critique pipeline...\n")
    result = app.invoke({"topic": "why the Groq LPU is fast for LLM inference"})

    print("\n" + "=" * 60)
    print("FINAL OUTPUT:\n")
    print(result["final"])


if __name__ == "__main__":
    main()
