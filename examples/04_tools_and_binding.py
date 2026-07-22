"""Lesson 04 — Tools and the manual tool-calling loop.

Before using a prebuilt agent, it helps to see the raw mechanics:
  1. Define tools with the @tool decorator.
  2. bind_tools() so the model knows they exist.
  3. The model responds with tool_calls (it doesn't run them — YOU do).
  4. Execute the tools, feed results back as ToolMessages, ask again.

Run:  uv run python examples/04_tools_and_binding.py
"""

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from agentlab.config import get_chat_model


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


TOOLS = {"add": add, "multiply": multiply}


def main() -> None:
    llm = get_chat_model()

    # The model now "knows" it can call add / multiply.
    llm_with_tools = llm.bind_tools(list(TOOLS.values()))

    messages = [HumanMessage(content="What is 12 * 7, and then add 5 to the result?")]

    # First pass: the model decides which tool(s) to call.
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    # Loop until the model stops asking for tools.
    while ai_msg.tool_calls:
        for call in ai_msg.tool_calls:
            tool_fn = TOOLS[call["name"]]
            result = tool_fn.invoke(call["args"])
            print(f"  ↳ called {call['name']}({call['args']}) = {result}")
            # Feed each result back, tagged with the originating call id.
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

    print("\nFinal answer:", ai_msg.content)


if __name__ == "__main__":
    main()
