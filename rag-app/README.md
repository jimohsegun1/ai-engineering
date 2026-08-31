# RAG App (LangChain)

A minimal Retrieval-Augmented Generation (RAG) pipeline built with LangChain, written as a
learning project. Everything lives in one file, `rag_pipeline.py`, with each of the six RAG
steps clearly commented:

1. Prepare input document
2. Chunking
3. Create embeddings
4. Store embeddings in a vector database
5. Similarity search
6. RAG pipeline (retrieval + generation)

## Stack

- **Document loading / chunking**: LangChain (`TextLoader`, `RecursiveCharacterTextSplitter`)
- **Embeddings**: local, free HuggingFace model (`sentence-transformers/all-MiniLM-L6-v2`) — no API key or cost
- **Vector database**: Chroma, persisted to `chroma_db/`
- **LLM (generation)**: [Qorebit](https://qorebit.ai) — an OpenAI-compatible API gateway, called via `langchain-openai`'s `ChatOpenAI` with a custom `base_url`

## Project structure

```
rag-app/
├── data/
│   └── sample.txt        # the input document the pipeline is built around
├── rag_pipeline.py        # the whole pipeline, steps 1-6
├── requirements.txt
├── .env                    # your real API key (gitignored, never commit this)
├── .env.example            # template showing which variable to set
└── chroma_db/              # generated on each run — the persisted vector store
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

### 5. Set up your API key

Copy the template into a real `.env` file:

**PowerShell:**
```powershell
Copy-Item .env.example .env
```

**Git Bash:**
```bash
cp .env.example .env
```

Then open `.env` and replace `your-api-key-here` with your real Qorebit API key (from your
Qorebit dashboard → API Keys, looks like `qb_live_...`).

`.env` is listed in `.gitignore` — it should never be committed. `.env.example` holds
only a placeholder and is safe to commit.

## Running it

Make sure your virtual environment is activated (prompt shows `(venv)`) and you're in the
`rag-app/` directory, then:

```powershell
python rag_pipeline.py
```

Each step prints its own output as it runs, so you can see exactly what's happening:

```
[1] Loaded 1 document(s) from data/sample.txt
[2] Split into 8 chunks
[3] Loaded embedding model: sentence-transformers/all-MiniLM-L6-v2
[4] Stored 8 chunks in 'chroma_db'
[5] Top 3 chunks for: '...'
    result 0: ...
[6] Question: ...
[6] Answer: ...
```

The first run downloads the embedding model (a few hundred MB) and caches it locally in
`~/.cache/huggingface` — later runs are fast.

## Trying your own questions

Open `rag_pipeline.py` and change the `QUESTION` constant near the top, then rerun the
script. You can also swap in your own document by replacing `data/sample.txt` (or changing
`DOCUMENT_PATH`) and adjusting `CHUNK_SIZE` / `CHUNK_OVERLAP` if needed.

## Notes / gotchas

- **"Failed to send telemetry event" warnings.** If you see these, it's a version
  mismatch between `chromadb` and a too-new `posthog` release (its `capture()` signature
  changed). `requirements.txt` pins `posthog<4` to avoid this — if it still shows up, run
  `pip install -r requirements.txt` again to make sure the pin took effect. It's harmless
  either way and never affects the pipeline's actual output.
- **Vector store resets on every run.** `rag_pipeline.py` deletes `chroma_db/` before
  rebuilding it, so re-running never duplicates chunks — it's not meant to persist across
  runs of a different document.
- **Custom headers for Qorebit.** The `ChatOpenAI` client is configured with a custom
  `User-Agent` header, because Qorebit's WAF blocks the default User-Agent string sent by
  the `openai` Python SDK. `HTTP-Referer` / `X-Title` are also sent, matching Qorebit's docs.
- **Embeddings and generation use different providers.** Qorebit's docs only cover chat
  completions, not embeddings, so step 3 uses a free local model instead of calling Qorebit
  for that step.
