# Semantic Caching

Part of the [AI Engineering From Scratch](../README.md) learning project — see the root
README for one-time setup (virtual environment, dependencies, API keys).

An exact-match cache (a plain dict keyed by the question string) only helps when someone
asks the *identical* question twice. Real users rephrase — "What's the capital of France?"
vs "What's the capital city of France?" are the same question to a person, but different
dict keys. `01_semantic_cache.py` fixes this: it embeds every question, and before calling
the LLM it checks a Chroma-backed cache for a *similar enough* past question. If one is
found within a distance threshold, it returns that cached answer instantly instead of
paying for another generation.

Runs on the same free local Hugging Face embeddings + `flan-t5-base` + Chroma stack as
[`rag-app/rag_pipeline_huggingface.py`](../rag-app/README.md) — no API key needed. The
threshold (`SIMILARITY_THRESHOLD = 0.3`) was calibrated empirically: true paraphrases of the
same question scored 0.09-0.15 on Chroma's squared-L2 distance, while a related-but-different
question scored 1.05 and an unrelated one scored 1.71 — 0.3 sits cleanly in the gap.

A fixed run of 5 questions (exact repeats, paraphrases, and genuinely new questions) prints
each as a HIT or MISS with its distance and wall-clock time, then a summary comparing average
hit vs. miss latency — cache hits were **~46x faster** than a live LLM call in a test run.

## Running it

```powershell
python 10-caching/01_semantic_cache.py
```
or `cd 10-caching` and run it by its bare name. Either way, the venv needs to be active and
`data/`/`db/` are always resolved relative to the project root.

## Notes / gotchas

- **`flan-t5-base` answered "london" for the capital of France.** It's the same small
  (~250M parameter) model used throughout this project, chosen for being free and
  CPU-friendly, not for accuracy — see the other concepts' READMEs for more examples of
  this. The interesting part for *this* demo: the cache faithfully serves whatever answer
  got stored on the first call, right or wrong. A semantic cache guarantees consistency
  (the same question always gets the same answer), not correctness — a wrong first answer
  stays wrong for every paraphrase after it, until the cache is cleared.
- **The similarity threshold is tuned for this project's embedding model and a handful of
  short factual questions.** A different embedding model, a different kind of question
  (longer, more technical), or a different vector space (Chroma defaults to squared L2,
  not cosine) would need its own calibration — don't reuse `0.3` blindly elsewhere.

See the root [README](../README.md) for shared notes (telemetry warnings, vector store
reset behavior, dependency version pins) that apply across the whole project.
