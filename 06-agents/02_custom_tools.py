"""Agents: defining custom tools.

Before an agent can use a tool, the tool itself has to be defined: the
@tool decorator turns an ordinary typed Python function into something an
LLM can be told about and can invoke. The function's docstring becomes the
tool's description (what the LLM reads to decide when to use it), and its
type hints become the input schema. No LLM involved in this file — just the
tool abstraction itself, which every other file in this folder builds on.
"""

from langchain_core.tools import tool

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())


@tool
def is_palindrome(text: str) -> bool:
    """Check whether a piece of text reads the same forwards and backwards, ignoring spaces and case."""
    cleaned = text.replace(" ", "").lower()
    return cleaned == cleaned[::-1]


print_step(1, "Inspect what @tool generated from each function")
for t in [word_count, is_palindrome]:
    print(f"\nname:        {t.name}")
    print(f"description: {t.description}")
    print(f"args schema: {t.args}")

print_step(2, "Invoke the tools directly, the same way an agent would")
print(f"word_count('the quick brown fox') -> {word_count.invoke({'text': 'the quick brown fox'})}")
print(f"is_palindrome('Was it a car or a cat I saw') -> "
      f"{is_palindrome.invoke({'text': 'Was it a car or a cat I saw'})}")
