"""Lesson 02 — Prompt templates + LCEL chains.

Instead of hand-building message lists every time, use a ChatPromptTemplate with
placeholders, then compose it with the model and an output parser using the LCEL
pipe operator:  prompt | model | parser

Run:  uv run python examples/02_prompts_and_chains.py
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agentlab.config import get_chat_model


def main() -> None:
    llm = get_chat_model()

    # {placeholders} get filled in at call time via .invoke({...}).
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a translator. Translate the text to {language}. Output only the translation."),
            ("human", "{text}"),
        ]
    )

    # StrOutputParser pulls the plain string out of the AIMessage.
    # The pipe (|) wires the three pieces into a single runnable "chain".
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"language": "French", "text": "Good morning, how are you?"})
    print("Translation:", result)

    # Chains support .batch() (parallel inputs) and .stream() (token streaming) too.
    print("\nStreaming a longer answer:\n")
    story_chain = (
        ChatPromptTemplate.from_template("Write a 2-sentence bedtime story about a {animal}.")
        | llm
        | StrOutputParser()
    )
    for chunk in story_chain.stream({"animal": "robot"}):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
