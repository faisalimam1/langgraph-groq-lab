"""Lesson 03 — Structured output with Pydantic.

Often you want typed data back, not free text. `with_structured_output(Schema)` makes
the model return an instance of your Pydantic model (validated for you). This is the
foundation for reliable tool arguments and downstream code.

Run:  uv run python examples/03_structured_output.py
"""

from pydantic import BaseModel, Field

from agentlab.config import get_chat_model


# Describe the shape you want. Field descriptions guide the model — treat them as prompts.
class Recipe(BaseModel):
    """A simple cooking recipe."""

    name: str = Field(description="The dish name")
    ingredients: list[str] = Field(description="List of ingredients with rough quantities")
    steps: list[str] = Field(description="Ordered preparation steps")
    minutes: int = Field(description="Approximate total time in minutes")


def main() -> None:
    llm = get_chat_model()

    # Bind the schema. The returned runnable outputs a Recipe object, not text.
    structured_llm = llm.with_structured_output(Recipe)

    recipe = structured_llm.invoke("Give me a quick recipe for a classic omelette.")

    print(f"🍳 {recipe.name}  (~{recipe.minutes} min)\n")
    print("Ingredients:")
    for item in recipe.ingredients:
        print(f"  - {item}")
    print("\nSteps:")
    for i, step in enumerate(recipe.steps, 1):
        print(f"  {i}. {step}")

    # It's a real Python object — attribute access, type-checked.
    print(f"\n(type: {type(recipe).__name__}, ingredient count: {len(recipe.ingredients)})")


if __name__ == "__main__":
    main()
