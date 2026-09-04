# RAG App (LangChain)

A minimal Retrieval-Augmented Generation (RAG) pipeline built with LangChain, written as a
learning project, with each of the six RAG steps clearly commented:

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

### Chunking method demos

`chunking_methods/` has six standalone files, each demonstrating a different chunking
(step 2) strategy — everything else about the pipeline stays the same. In every file, step
6 (the Qorebit call) is **commented out on purpose**, so you can read/run steps 1-5 for
free and uncomment step 6 yourself when you're ready to spend Qorebit credits:

| File | Method | Input document | What it shows |
| --- | --- | --- | --- |
| `01_character_splitter.py` | `CharacterTextSplitter` | `data/sample.txt` | Fixed-size, single-separator splitting — an oversized paragraph is *not* split further |
| `02_recursive_character_splitter.py` | `RecursiveCharacterTextSplitter` | `data/sample.txt` | Falls back through smaller separators to still hit the target size (same method `rag_pipeline.py` uses) |
| `03_token_splitter.py` | `TokenTextSplitter` | `data/sample.txt` | Sizes chunks by token count instead of character count |
| `04_markdown_header_splitter.py` | `MarkdownHeaderTextSplitter` | `data/sample.md` | Splits along `#`/`##` headers, keeping the heading path as metadata |
| `05_semantic_chunker.py` | `SemanticChunker` (langchain-experimental) | `data/sample.txt` | Splits where meaning shifts between sentences, using embeddings rather than a fixed size |
| `06_code_splitter.py` | `RecursiveCharacterTextSplitter.from_language(PYTHON)` | `data/sample_code.py` | Splits source code along function/class boundaries instead of prose separators |

Every file resolves its input/output paths from its own location, not the current working
directory, so you can run them either from `rag-app/`:
```powershell
python chunking_methods/01_character_splitter.py
python chunking_methods/02_recursive_character_splitter.py
python chunking_methods/03_token_splitter.py
python chunking_methods/04_markdown_header_splitter.py
python chunking_methods/05_semantic_chunker.py
python chunking_methods/06_code_splitter.py
```
or from inside `chunking_methods/` itself:
```powershell
cd chunking_methods
python 01_character_splitter.py
```
Either way, the venv still needs to be active and `data/`/`db/` are always resolved
relative to `rag-app/`, never relative to `chunking_methods/`.

### PDF input demo

`rag_pipeline_pdf.py` is the same six-step pipeline as `rag_pipeline.py`, but the input
document is `data/sample.pdf` (a small 3-page PDF about vector databases) instead of a
`.txt` file, loaded with `PyPDFLoader` instead of `TextLoader`. The key difference to
notice: `PyPDFLoader` returns **one Document per PDF page** (each carrying a `page` number
in its metadata) rather than a single Document for the whole file, so step 1 already
produces multiple documents before chunking even runs — every chunk downstream also keeps
track of which page it came from. Like the other files, step 6 (the Qorebit call) is
commented out by default.

```powershell
python rag_pipeline_pdf.py
```

## Stack

- **Document loading / chunking**: LangChain (`TextLoader` or `PyPDFLoader`, `RecursiveCharacterTextSplitter`)
- **Embeddings**: local, free HuggingFace model (`sentence-transformers/all-MiniLM-L6-v2`) — no API key or cost
- **Vector database**: Chroma — every file persists under `db/`, each to its own
  subfolder (e.g. `db/qorebit/`, `db/huggingface_local/`, `db/chunking_code/`) so running
  one never clobbers another's store
- **LLM (generation)**: Qorebit, a local `flan-t5-base` model, or Hugging Face's hosted
  Inference API — see the table above for which file uses which

## Project structure

```
rag-app/
├── data/
│   ├── sample.txt                        # the input document most files use
│   ├── sample.md                         # same content, restructured with Markdown headers
│   ├── sample_code.py                    # small Python module, for the code-aware splitter
│   └── sample.pdf                        # small 3-page PDF, for rag_pipeline_pdf.py
├── chunking_methods/                     # six chunking-method demos, see table above
│   ├── 01_character_splitter.py
│   ├── 02_recursive_character_splitter.py
│   ├── 03_token_splitter.py
│   ├── 04_markdown_header_splitter.py
│   ├── 05_semantic_chunker.py
│   └── 06_code_splitter.py
├── rag_pipeline.py                         # Qorebit version, steps 1-6
├── rag_pipeline_huggingface.py             # fully local version, steps 1-6
├── rag_pipeline_huggingface_hosted.py      # HF hosted Inference API version, steps 1-6
├── rag_pipeline_pdf.py                     # PDF input version, steps 1-6
├── requirements.txt                         # covers every file above
├── .env                                     # your real API keys (gitignored, never commit this)
├── .env.example                             # template showing which variables to set
└── db/                                      # generated on every run, gitignored
    ├── qorebit/                               # from rag_pipeline.py
    ├── huggingface_local/                     # from rag_pipeline_huggingface.py
    ├── huggingface_hosted/                    # from rag_pipeline_huggingface_hosted.py
    ├── pdf/                                   # from rag_pipeline_pdf.py
    └── chunking_<method>/                     # one per chunking_methods/ file
```

## Setup

### 1. Open a terminal in the project folder

Navigate to `rag-app/` — every command below assumes you're standing in this directory.

**PowerShell:**
```powershell
cd C:\Users\jimoh\OneDrive\Desktop\OFFICE\ai-engineering\rag-app
```

**Git Bash:**
```bash
cd /c/Users/jimoh/OneDrive/Desktop/OFFICE/ai-engineering/rag-app
```

### 2. Create the virtual environment

A virtual environment keeps this project's dependencies separate from your system Python.
You only need to do this once — it creates a `venv/` folder here.

```powershell
python -m venv venv
```

### 3. Activate the virtual environment

You must activate it in **every new terminal session** before running or installing
anything for this project. When it's active, your prompt line starts with `(venv)`.

**PowerShell:**
```powershell
venv\Scripts\Activate.ps1
```

If PowerShell blocks this with an execution-policy error, run this once (allows locally
created scripts to run for your user account) and then retry:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**cmd.exe:**
```cmd
venv\Scripts\activate.bat
```

**Git Bash:**
```bash
source venv/Scripts/activate
```

To leave the virtual environment later, run `deactivate` in any shell.

### 4. Install dependencies

With `(venv)` showing in your prompt, install everything from `requirements.txt` — never
install packages one-off, always add them to `requirements.txt` first:

```powershell
pip install -r requirements.txt
```

This is the slowest step the first time (a few minutes) since it downloads the embedding
model's dependencies (PyTorch, etc.). Later installs are fast.

### 5. Set up your API keys (only needed for two of the three files)

`rag_pipeline_huggingface.py` needs no API key at all — skip this step if that's the only
file you plan to run.

Copy the template into a real `.env` file:

**PowerShell:**
```powershell
Copy-Item .env.example .env
```

**Git Bash:**
```bash
cp .env.example .env
```

Then open `.env` and fill in whichever key(s) you need:
- `QOREBIT_API_KEY` — for `rag_pipeline.py`. From your Qorebit dashboard → API Keys, looks
  like `qb_live_...`.
- `HUGGINGFACEHUB_API_TOKEN` — for `rag_pipeline_huggingface_hosted.py`. Get a free one at
  [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → "Create new
  token" → Read access is enough.

`.env` is listed in `.gitignore` — it should never be committed. `.env.example` holds
only placeholders and is safe to commit.

## Running it

Make sure your virtual environment is activated (prompt shows `(venv)`) and you're in the
`rag-app/` directory, then run whichever version you want:

```powershell
python rag_pipeline.py                          # Qorebit — needs QOREBIT_API_KEY in .env
python rag_pipeline_huggingface.py              # fully local — no setup needed beyond step 4
python rag_pipeline_huggingface_hosted.py       # HF hosted API — needs HUGGINGFACEHUB_API_TOKEN in .env
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
changing `DOCUMENT_PATH`) and adjusting `CHUNK_SIZE` / `CHUNK_OVERLAP` if needed — do this
in all three files if you want them all to use the same document.

## Notes / gotchas

- **"Failed to send telemetry event" warnings.** If you see these, it's a version
  mismatch between `chromadb` and a too-new `posthog` release (its `capture()` signature
  changed). `requirements.txt` pins `posthog<4` to avoid this — if it still shows up, run
  `pip install -r requirements.txt` again to make sure the pin took effect. It's harmless
  either way and never affects the pipeline's actual output.
- **Vector store resets on every run.** Every script deletes its own persisted Chroma
  folder under `db/` before rebuilding it, so re-running never duplicates chunks — it's not
  meant to persist across runs of a different document.
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
- **`05_semantic_chunker.py`'s default threshold is tuned for long documents.**
  `SemanticChunker`'s default `breakpoint_threshold_amount` (95th percentile) only treats
  the single most extreme meaning-shift as a split point, which collapses a short document
  like our ~10-sentence sample into one giant chunk. The file lowers it to 50 so it finds
  multiple breakpoints instead — on a longer document you'd likely want it closer to the
  default. It can also emit an empty trailing chunk, which the file filters out.
