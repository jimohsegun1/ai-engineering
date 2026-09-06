"""Conversation memory: VectorStoreRetrieverMemory.

Every exchange is embedded and stored in Chroma, then load_memory_variables
retrieves whichever *past* exchanges are semantically closest to the new
input — not just the most recent ones like the other memory types in this
folder. No LLM needed, only the same free local embedding model used
throughout this project.
"""

import os
import shutil
import textwrap
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain.memory import VectorStoreRetrieverMemory
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "memory_vectorstore"
TOP_K = 1

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


def print_passage(label: str, text: str) -> None:
    print(f"\n  {label}")
    print("  " + "-" * 60)
    wrapped = textwrap.fill(text, width=68, initial_indent="  ", subsequent_indent="  ")
    print(wrapped)


PAST_TURNS = [
    ("My favorite color is blue.", "Got it, blue is your favorite color."),
    ("I have a dog named Rex.", "Nice! Rex sounds like a good dog."),
    ("I'm planning a trip to Japan next year.", "That sounds exciting! Japan has a lot to offer."),
]

NEW_INPUT = "What's my dog's name again?"

print_step(1, "Store past exchanges in the vector store")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma(embedding_function=embeddings, persist_directory=str(PERSIST_DIRECTORY))
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
memory = VectorStoreRetrieverMemory(retriever=retriever)

for user_input, ai_output in PAST_TURNS:
    memory.save_context({"input": user_input}, {"output": ai_output})
    print(f"Saved: {user_input!r} -> {ai_output!r}")


print_step(2, "Retrieve the most relevant past exchange for a new input")
print(f"New input: {NEW_INPUT!r}")
retrieved = memory.load_memory_variables({"input": NEW_INPUT})["history"]
print_passage(f"Top {TOP_K} relevant memory", retrieved)
