"""Lesson 06 — Short-term memory with a checkpointer.

By default an agent forgets everything between .invoke() calls. Attach a checkpointer
and pass a `thread_id`, and LangGraph persists the conversation state — so the agent
remembers earlier turns within the same thread.

Run:  uv run python examples/06_memory_checkpointer.py
"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from agentlab.config import get_chat_model


def main() -> None:
    llm = get_chat_model()

    # InMemorySaver keeps state in RAM (great for demos). Swap for SqliteSaver/Postgres
    # in production to persist across restarts.
    checkpointer = InMemorySaver()

    # No tools here — just showing memory. The agent still works fine tool-less.
    agent = create_agent(model=llm, tools=[], checkpointer=checkpointer)

    # The thread_id ties turns together. Same id = same conversation.
    config = {"configurable": {"thread_id": "demo-conversation-1"}}

    def ask(text: str) -> None:
        result = agent.invoke({"messages": [("user", text)]}, config=config)
        print(f"You:   {text}")
        print(f"Agent: {result['messages'][-1].content}\n")

    ask("Hi! My name is Sania and my favorite color is teal.")
    ask("What's my name?")
    ask("And what's my favorite color?")

    # Different thread_id = fresh memory. The agent won't know the name here.
    print("--- switching to a NEW thread (memory is per-thread) ---\n")
    other = {"configurable": {"thread_id": "demo-conversation-2"}}
    result = agent.invoke({"messages": [("user", "Do you know my name?")]}, config=other)
    print(f"You:   Do you know my name?")
    print(f"Agent: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
