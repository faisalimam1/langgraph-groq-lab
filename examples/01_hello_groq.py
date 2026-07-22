"""Lesson 01 — Hello, Groq.

The smallest possible LangChain program: send messages to a Groq chat model and
print the reply. This confirms your key + model are working.

Run:  uv run python examples/01_hello_groq.py
"""

from langchain_core.messages import HumanMessage, SystemMessage

from agentlab.config import get_chat_model


def main() -> None:
    # get_chat_model() returns a ChatGroq instance wired to your GROQ_API_KEY.
    llm = get_chat_model()

    # A chat model takes a list of messages. SystemMessage sets behavior;
    # HumanMessage is the user's turn.
    messages = [
        SystemMessage(content="You are a concise assistant. Answer in one sentence."),
        HumanMessage(content="What is LangChain, and why pair it with Groq?"),
    ]

    # .invoke() runs the model once and returns an AIMessage.
    response = llm.invoke(messages)

    print("Model:", llm.model_name)
    print("\nReply:\n", response.content)

    # Token usage is attached as metadata — handy for cost/latency awareness.
    usage = response.usage_metadata
    if usage:
        print(f"\nTokens — in: {usage['input_tokens']}, out: {usage['output_tokens']}")


if __name__ == "__main__":
    main()
