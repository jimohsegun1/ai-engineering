"""Conversation memory: ConversationBufferMemory.

Keeps the entire conversation transcript verbatim, so it keeps growing on
every turn — simplest possible memory, but eventually blows past a model's
context window on a long-running conversation.
"""

from langchain.memory import ConversationBufferMemory

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


TURNS = [
    ("What's the capital of France?", "The capital of France is Paris."),
    ("What's a famous landmark there?", "The Eiffel Tower is a famous landmark in Paris."),
    ("How tall is it?", "The Eiffel Tower is about 330 meters tall."),
]

memory = ConversationBufferMemory()

for i, (user_input, ai_output) in enumerate(TURNS, start=1):
    print_step(i, f"Turn {i}")
    memory.save_context({"input": user_input}, {"output": ai_output})
    print(f"User: {user_input}")
    print(f"AI:   {ai_output}")
    print(f"\nFull buffer so far ({len(memory.buffer)} chars):")
    print(memory.load_memory_variables({})["history"])
