"""Semantic caching: skip repeat LLM calls for near-duplicate questions.

An exact-match cache (a plain dict keyed by the question string) only helps
when someone asks the *identical* question twice. Real users rephrase —
"What's the capital of France?" vs "What's the capital city of France?" are
the same question to a person, but different dict keys. A semantic cache
fixes this: it embeds every question, and before calling the LLM it checks
whether a *similar enough* question was already answered. If so, it returns
that cached answer instantly instead of paying for another generation.

This reuses the same free local Hugging Face embeddings + Chroma stack as
every other demo in this project (`sentence-transformers/all-MiniLM-L6-v2`)
and `google/flan-t5-base` for generation — no API key needed.
"""

import os
import shutil
import time
from pathlib import Path

# Must be set before chromadb is imported.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "google/flan-t5-base"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "semantic_cache"

# Chroma's similarity_search_with_score returns squared L2 distance by
# default (lower = more similar). Calibrated empirically: true paraphrases
# of the same question scored 0.09-0.15 here, while a related-but-different
# question ("capital of Japan" vs "capital of France") scored 1.05 and a
# wholly unrelated one scored 1.71 — 0.3 sits cleanly in the gap between them.
SIMILARITY_THRESHOLD = 0.3

# Includes exact repeats, paraphrases of earlier questions, and genuinely
# new ones, so both cache hits and misses show up in the run.
QUESTIONS = [
    "What is the capital of France?",
    "What's the capital city of France?",
    "What is the capital of Japan?",
    "Tell me France's capital.",
    "What is the capital of Japan?",
]


# --- Step 1: Set up the cache store and the LLM ---
print_step(1, "Set up the cache store and the LLM")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)  # avoid duplicating cache entries on rerun
cache_store = Chroma(embedding_function=embeddings, persist_directory=str(PERSIST_DIRECTORY))
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 50},
)
print(f"Cache store ready at '{PERSIST_DIRECTORY}' (empty)")


def nearest_cache_entry(question: str) -> tuple[str, float] | None:
    if cache_store._collection.count() == 0:
        return None
    doc, distance = cache_store.similarity_search_with_score(question, k=1)[0]
    return doc.metadata["answer"], distance


def cache_save(question: str, answer: str) -> None:
    cache_store.add_texts([question], metadatas=[{"answer": answer}])


def answer_question(question: str) -> tuple[str, bool, float, float | None]:
    start = time.perf_counter()
    nearest = nearest_cache_entry(question)
    if nearest is not None and nearest[1] <= SIMILARITY_THRESHOLD:
        answer, distance = nearest
        return answer, True, time.perf_counter() - start, distance

    answer = llm.invoke(f"Answer briefly: {question}").strip()
    cache_save(question, answer)
    # Distance to the nearest (too-far) entry, if any, shown for comparison
    # against SIMILARITY_THRESHOLD — makes clear why this was a miss.
    distance = nearest[1] if nearest is not None else None
    return answer, False, time.perf_counter() - start, distance


# --- Step 2: Run each question through the cache ---
print_step(2, f"Run {len(QUESTIONS)} questions through the cache")
hit_times: list[float] = []
miss_times: list[float] = []
for question in QUESTIONS:
    answer, hit, elapsed, distance = answer_question(question)
    status = "HIT " if hit else "MISS"
    distance_note = f", distance={distance:.4f}" if distance is not None else ""
    print(f"\n[{status}] {elapsed:.3f}s{distance_note}")
    print(f"  Question: {question}")
    print(f"  Answer:   {answer}")
    (hit_times if hit else miss_times).append(elapsed)


# --- Step 3: Summary ---
print_step(3, "Summary")
print(f"Hits:  {len(hit_times)}/{len(QUESTIONS)}")
print(f"Misses: {len(miss_times)}/{len(QUESTIONS)}")
if hit_times and miss_times:
    avg_hit = sum(hit_times) / len(hit_times)
    avg_miss = sum(miss_times) / len(miss_times)
    speedup = avg_miss / avg_hit if avg_hit else float("inf")
    print(f"Average hit time:  {avg_hit:.3f}s")
    print(f"Average miss time: {avg_miss:.3f}s")
    print(f"Cache hits were about {speedup:.0f}x faster than a live LLM call.")
