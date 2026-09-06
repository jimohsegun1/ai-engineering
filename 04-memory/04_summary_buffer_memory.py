"""Conversation memory: ConversationSummaryBufferMemory.

A hybrid: recent exchanges are kept verbatim, but once the buffer grows past
max_token_limit, the oldest ones are rolled into a running summary instead of
being dropped outright — a middle ground between buffer and summary memory.
"""

from langchain.memory import ConversationSummaryBufferMemory
from langchain_huggingface import HuggingFacePipeline

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


CHAT_MODEL = "google/flan-t5-base"
MAX_TOKEN_LIMIT = 40

TURNS = [
    ("What's the capital of France?", "The capital of France is Paris."),
    ("What's a famous landmark there?", "The Eiffel Tower is a famous landmark in Paris."),
    ("How tall is it?", "The Eiffel Tower is about 330 meters tall."),
    ("Who built it?", "It was designed and built by Gustave Eiffel's engineering company."),
]

llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 200},
)
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=MAX_TOKEN_LIMIT)

for i, (user_input, ai_output) in enumerate(TURNS, start=1):
    print_step(i, f"Turn {i}")
    memory.save_context({"input": user_input}, {"output": ai_output})
    print(f"User: {user_input}")
    print(f"AI:   {ai_output}")
    print(f"\nMemory so far (summary for anything past the last {MAX_TOKEN_LIMIT} tokens):")
    print(memory.load_memory_variables({})["history"])
