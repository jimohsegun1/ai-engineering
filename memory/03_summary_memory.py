"""Conversation memory: ConversationSummaryMemory.

Instead of storing the transcript verbatim, an LLM rewrites it into a running
summary after every turn — stays small no matter how long the conversation
runs, at the cost of losing exact wording. Runs on a local, free Hugging Face
model, same as rag_pipeline_huggingface.py.
"""

from langchain.memory import ConversationSummaryMemory
from langchain_huggingface import HuggingFacePipeline

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


CHAT_MODEL = "google/flan-t5-base"

TURNS = [
    ("What's the capital of France?", "The capital of France is Paris."),
    ("What's a famous landmark there?", "The Eiffel Tower is a famous landmark in Paris."),
    ("How tall is it?", "The Eiffel Tower is about 330 meters tall."),
]

llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 200},
)
memory = ConversationSummaryMemory(llm=llm)

for i, (user_input, ai_output) in enumerate(TURNS, start=1):
    print_step(i, f"Turn {i}")
    memory.save_context({"input": user_input}, {"output": ai_output})
    print(f"User: {user_input}")
    print(f"AI:   {ai_output}")
    print("\nRunning summary (replaces the raw transcript):")
    print(memory.load_memory_variables({})["history"])
