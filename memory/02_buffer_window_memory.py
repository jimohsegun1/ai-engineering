"""Conversation memory: ConversationBufferWindowMemory.

Only keeps the last k exchanges verbatim, dropping older ones — bounds
memory size at the cost of forgetting anything outside the window.
"""

from langchain.memory import ConversationBufferWindowMemory

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


WINDOW_SIZE = 2

TURNS = [
    ("What's the capital of France?", "The capital of France is Paris."),
    ("What's a famous landmark there?", "The Eiffel Tower is a famous landmark in Paris."),
    ("How tall is it?", "The Eiffel Tower is about 330 meters tall."),
    ("Who built it?", "It was designed and built by Gustave Eiffel's engineering company."),
]

memory = ConversationBufferWindowMemory(k=WINDOW_SIZE)

for i, (user_input, ai_output) in enumerate(TURNS, start=1):
    print_step(i, f"Turn {i}")
    memory.save_context({"input": user_input}, {"output": ai_output})
    print(f"User: {user_input}")
    print(f"AI:   {ai_output}")
    print(f"\nWindow (last {WINDOW_SIZE} exchanges only):")
    print(memory.load_memory_variables({})["history"])
