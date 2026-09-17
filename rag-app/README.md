# RAG Pipeline

Part of the [AI Engineering From Scratch](../README.md) learning project — see the root
README for one-time setup (virtual environment, dependencies, API keys).

A minimal Retrieval-Augmented Generation (RAG) pipeline, with each of the six RAG steps
clearly commented:

1. Prepare input document
2. Chunking
3. Create embeddings
4. Store embeddings in a vector database
5. Similarity search
6. RAG pipeline (retrieval + generation)

There are **three versions of the pipeline** — same steps, same structure, different
generation backend:

| File | Generation (step 6) | Setup needed |
| --- | --- | --- |
| `rag_pipeline.py` | [Qorebit](https://qorebit.ai) — a hosted, OpenAI-compatible API | Qorebit API key in `.env` |
| `rag_pipeline_huggingface.py` | `google/flan-t5-base`, run locally via `transformers` | None — no API key, no internet-dependent call |
| `rag_pipeline_huggingface_hosted.py` | A larger model (`HuggingFaceH4/zephyr-7b-beta`) via Hugging Face's hosted Inference API | Hugging Face access token in `.env` |

All three use the same free local Hugging Face model for embeddings (step 3):
`sentence-transformers/all-MiniLM-L6-v2`.

## PDF input demo

`rag_pipeline_pdf.py` is the same six-step pipeline as `rag_pipeline.py`, but the input
document is `data/sample.pdf` (a small 3-page PDF about vector databases) instead of a
`.txt` file, loaded with `PyPDFLoader` instead of `TextLoader`. The key difference to
notice: `PyPDFLoader` returns **one Document per PDF page** (each carrying a `page` number
in its metadata) rather than a single Document for the whole file, so step 1 already
produces multiple documents before chunking even runs — every chunk downstream also keeps
track of which page it came from. Like the other files, step 6 (the Qorebit call) is
commented out by default.

## Running it

Make sure your virtual environment is activated (prompt shows `(venv)`) and you're in the
project root, then run whichever version you want:

```powershell
python rag-app/rag_pipeline.py                          # Qorebit — needs QOREBIT_API_KEY in .env
python rag-app/rag_pipeline_huggingface.py              # fully local — no setup needed beyond step 4
python rag-app/rag_pipeline_huggingface_hosted.py       # HF hosted API — needs HUGGINGFACEHUB_API_TOKEN in .env
python rag-app/rag_pipeline_pdf.py                      # PDF input — needs QOREBIT_API_KEY in .env
```

Each step prints its own clearly-labeled section as it runs, so you can see exactly what's
happening — the chunks produced, what got stored, which passages matched your question, and
finally the generated answer.

The first run of any script downloads its models (a few hundred MB for embeddings, plus
~930MB more for `flan-t5-base` if you run the fully-local Hugging Face version) and caches
them locally in `~/.cache/huggingface` — later runs are fast, since nothing needs to be
re-downloaded. `rag_pipeline_huggingface_hosted.py` doesn't download a generation model at
all, since that model runs on Hugging Face's servers, not yours.

## Trying your own questions

Open whichever file you're using and change the `QUESTION` constant near the top, then
rerun it. You can also swap in your own document by replacing `data/sample.txt` (or
changing `DOCUMENT_PATH`) and adjusting `CHUNK_SIZE` / `CHUNK_OVERLAP` if needed — repeat the
change in every file you want to use the same document, since each one sets these constants
independently.

## Notes / gotchas

- **Custom headers for Qorebit.** In `rag_pipeline.py`, the `ChatOpenAI` client is
  configured with a custom `User-Agent` header, because Qorebit's WAF blocks the default
  User-Agent string sent by the `openai` Python SDK. `HTTP-Referer` / `X-Title` are also
  sent, matching Qorebit's docs.
- **Embeddings and generation use different providers in `rag_pipeline.py`.** Qorebit's
  docs only cover chat completions, not embeddings, so step 3 uses a free local model
  instead of calling Qorebit for that step.
- **`rag_pipeline_huggingface.py`'s answers are noticeably weaker.** `google/flan-t5-base`
  is a small (~250M parameter) model chosen so it runs on CPU with no GPU and no API key.
  Its answers lean extractive (echoing context almost verbatim, sometimes truncated) rather
  than genuinely composing a response. Swap `CHAT_MODEL` for a larger instruction-tuned
  model if you have the hardware and want better quality.
- **`rag_pipeline_huggingface_hosted.py` needs a fine-grained token.** A basic "Read"
  access token isn't enough — create one at
  [huggingface.co/settings/tokens/new?tokenType=fineGrained](https://huggingface.co/settings/tokens/new?tokenType=fineGrained)
  with the **"Make calls to Inference Providers"** permission checked, or every request
  fails with a 403 ("This authentication method does not have sufficient permissions...").
- **`rag_pipeline_huggingface_hosted.py` depends on Hugging Face's Inference API
  availability and quota.** Unlike the other two files, this one calls a remote service
  (here, routed to the `featherless-ai` provider), so it can hit the same class of issue we
  saw with other hosted providers:
  - Occasional `503 "temporarily at capacity"` errors for a given model — the script
    retries automatically a few times before giving up.
  - Free accounts get a small monthly credit allowance for Inference Providers; once it's
    used up you'll get `402 Payment Required` until it resets next month, or until you add
    pre-paid credits / a PRO subscription at huggingface.co/settings/billing.
  - If `CHAT_MODEL` itself stops being available, swap it for another instruction-tuned
    model that supports Hugging Face's hosted inference.

See the root [README](../README.md) for shared notes (telemetry warnings, vector store
reset behavior, dependency version pins) that apply across the whole project.
