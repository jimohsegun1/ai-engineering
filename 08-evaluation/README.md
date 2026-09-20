# RAG Evaluation

Part of the [AI Engineering From Scratch](../README.md) learning project — see the root
README for one-time setup (virtual environment, dependencies, API keys).

Every earlier RAG demo prints one answer to one question and you eyeball whether it looks
right. `01_rag_eval.py` replaces eyeballing with a small, fixed eval set — five questions
about `data/sample.txt`, each paired with keywords the correct answer should contain —
checked automatically every run.

It scores two layers separately: **retrieval** (did the vector store's top-`k` chunks contain
the expected keywords, independent of the LLM?) and **answer** (did the final generated answer
contain them too?). Splitting the two makes a failure diagnosable — a retrieval miss points at
chunking/embedding/search, while a retrieval hit with an answer miss points at the prompt or
the generation model instead. Runs entirely on the same free local Hugging Face embeddings +
`flan-t5-base` + Chroma stack as [`rag-app/rag_pipeline_huggingface.py`](../rag-app/README.md)
— no API key needed.

## Running it

```powershell
python 08-evaluation/01_rag_eval.py
```
or `cd 08-evaluation` and run it by its bare name. Either way, the venv needs to be active
and `data/`/`db/` are always resolved relative to the project root.

See the root [README](../README.md) for shared notes (telemetry warnings, vector store
reset behavior, dependency version pins) that apply across the whole project.
