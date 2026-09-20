# AI Engineering From Scratch

A collection of small, standalone LangChain demos written as a learning project — a RAG
pipeline plus focused demo folders for document loaders, chunking methods, conversation
memory, chain composition, prompting techniques, agents, LangGraph, RAG evaluation, and
deployment. Every file runs on its own; none of the demo folders depend on each other.

## Concepts

Each concept has its own README with the full write-up (what it demonstrates, a per-file
table, how to run it, and its own gotchas). This root README covers what's shared across all
of them: the stack, project layout, one-time setup, and cross-cutting notes.

| Concept | Folder |
| --- | --- |
| RAG pipeline | [`rag-app/`](rag-app/README.md) |
| Chunking methods | [`01-chunking-methods/`](01-chunking-methods/README.md) |
| Prompt engineering | [`02-prompt-engineering/`](02-prompt-engineering/README.md) |
| Document loaders | [`03-document-loaders/`](03-document-loaders/README.md) |
| Conversation memory | [`04-memory/`](04-memory/README.md) |
| Chain composition | [`05-chains/`](05-chains/README.md) |
| Agents | [`06-agents/`](06-agents/README.md) |
| LangGraph | [`07-langgraph/`](07-langgraph/README.md) |

## RAG evaluation demo

Every earlier RAG demo prints one answer to one question and you eyeball whether it looks
right. `08-evaluation/01_rag_eval.py` replaces eyeballing with a small, fixed eval set — five
questions about `data/sample.txt`, each paired with keywords the correct answer should
contain — checked automatically every run.

It scores two layers separately: **retrieval** (did the vector store's top-`k` chunks contain
the expected keywords, independent of the LLM?) and **answer** (did the final generated answer
contain them too?). Splitting the two makes a failure diagnosable — a retrieval miss points at
chunking/embedding/search, while a retrieval hit with an answer miss points at the prompt or
the generation model instead. Runs entirely on the same free local Hugging Face embeddings +
`flan-t5-base` + Chroma stack as `rag-app/rag_pipeline_huggingface.py` — no API key needed.

## Deployment demo

Every earlier demo calls `graph.invoke(...)` or `chain.invoke(...)` once, in-process, then
exits. `09-deployment/01_fastapi_agent_service.py` takes the supervisor graph from
`07-langgraph/07_supervisor_agent.py` and wraps it in a small [FastAPI](https://fastapi.tiangolo.com/)
app instead, so it runs as a long-lived HTTP service other programs can call:

- `GET /health` — a plain liveness check
- `POST /chat` — `{"question": "..."}` in, `{"route": "...", "answer": "..."}` out

The graph (and its connection to Ollama) is built once at server startup rather than once per
request. Needs the same Ollama setup as [`07-langgraph/`](07-langgraph/README.md) — no API key.

Run it, then call it from another terminal:

```powershell
python 09-deployment/01_fastapi_agent_service.py
```

```powershell
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\": \"What is 24 times 7, plus 10?\"}"
```

## Stack

- **Document loading / chunking**: LangChain (`TextLoader` or `PyPDFLoader`, `RecursiveCharacterTextSplitter`)
- **Embeddings**: local, free HuggingFace model (`sentence-transformers/all-MiniLM-L6-v2`) — no API key or cost
- **Vector database**: Chroma — every file persists under `db/`, each to its own
  subfolder (e.g. `db/qorebit/`, `db/huggingface_local/`, `db/chunking_code/`) so running
  one never clobbers another's store
- **LLM (generation)**: Qorebit, a local `flan-t5-base` model, Hugging Face's hosted
  Inference API, or Ollama (`llama3.2:3b`, local) for `06-agents/` and `07-langgraph/` — see
  the tables above for which file uses which

## Project structure

Shared resources (`data/`, `db/`, `venv/`, config files) live at the project root, so any
future learning module in this repo can reuse them too. `rag-app/` holds only the four
pipeline files themselves; every other technique gets its own top-level demo folder.

```
ai-engineering/                             # project root
├── data/
│   ├── sample.txt                        # the input document most files use
│   ├── sample2.txt                       # a second doc, for the directory-loader demo
│   ├── sample.md                         # same content as sample.txt, restructured with Markdown headers
│   ├── sample_code.py                    # small Python module, for the code-aware splitter
│   ├── sample.pdf                        # small 3-page PDF, for rag_pipeline_pdf.py / the PDF loader demo
│   ├── sample.csv                        # small table, for the CSV loader demo
│   └── sample.json                       # small nested JSON file, for the JSON loader demo
├── 01-chunking-methods/                     # six chunking-method demos, see its README
│   ├── README.md
│   ├── 01_character_splitter.py
│   ├── 02_recursive_character_splitter.py
│   ├── 03_token_splitter.py
│   ├── 04_markdown_header_splitter.py
│   ├── 05_semantic_chunker.py
│   └── 06_code_splitter.py
├── 03-document-loaders/                      # six document-loader demos, see its README
│   ├── README.md
│   ├── 01_text_loader.py
│   ├── 02_pdf_loader.py
│   ├── 03_csv_loader.py
│   ├── 04_json_loader.py
│   ├── 05_directory_loader.py
│   └── 06_web_loader.py
├── 04-memory/                             # five conversation-memory demos, see its README
│   ├── README.md
│   ├── 01_buffer_memory.py
│   ├── 02_buffer_window_memory.py
│   ├── 03_summary_memory.py
│   ├── 04_summary_buffer_memory.py
│   └── 05_vectorstore_retriever_memory.py
├── 05-chains/                             # eleven chain-composition demos, see its README
│   ├── README.md
│   ├── 01_simple_chain.py
│   ├── 02_sequential_chain.py
│   ├── 03_parallel_chain.py
│   ├── 04_router_chain.py
│   ├── 05_transform_chain.py
│   ├── 06_stuff_documents_chain.py
│   ├── 07_map_reduce_chain.py
│   ├── 08_refine_chain.py
│   ├── 09_retrieval_chain.py
│   ├── 10_math_chain.py
│   └── 11_conversational_retrieval_chain.py
├── 02-prompt-engineering/                    # five prompting-technique demos, see its README
│   ├── README.md
│   ├── 01_zero_shot_prompting.py
│   ├── 02_few_shot_prompting.py
│   ├── 03_chain_of_thought_prompting.py
│   ├── 04_role_based_prompting.py
│   └── 05_structured_output_prompting.py
├── 06-agents/                             # nine agent demos, see its README
│   ├── README.md
│   ├── 01_react_agent.py                     # Ollama
│   ├── 01_react_agent_qorebit.py             # Qorebit
│   ├── 02_custom_tools.py                    # no LLM
│   ├── 03_tool_calling_agent.py               # Ollama
│   ├── 03_tool_calling_agent_qorebit.py       # Qorebit
│   ├── 04_retriever_tool_agent.py             # Ollama
│   ├── 04_retriever_tool_agent_qorebit.py     # Qorebit
│   ├── 05_multi_tool_agent.py                 # Ollama
│   └── 05_multi_tool_agent_qorebit.py         # Qorebit
├── 07-langgraph/                          # twelve LangGraph demos, see its README (needs Ollama)
│   ├── README.md
│   ├── 01_simple_graph.py
│   ├── 02_conditional_graph.py
│   ├── 03_tool_calling_agent.py
│   ├── 03_tool_calling_agent_qorebit.py
│   ├── 04_persistent_memory.py
│   ├── 04_persistent_memory_qorebit.py
│   ├── 05_multi_agent_graph.py
│   ├── 05_multi_agent_graph_qorebit.py
│   ├── 06_streaming.py
│   ├── 06_streaming_qorebit.py
│   ├── 07_supervisor_agent.py
│   └── 07_supervisor_agent_qorebit.py
├── 08-evaluation/                          # RAG eval harness, see section above
│   └── 01_rag_eval.py
├── 09-deployment/                          # FastAPI agent service, see section above
│   └── 01_fastapi_agent_service.py           # needs Ollama
├── rag-app/
│   ├── README.md                           # concept write-up: setup, running it, gotchas
│   ├── rag_pipeline.py                     # Qorebit version, steps 1-6
│   ├── rag_pipeline_huggingface.py         # fully local version, steps 1-6
│   ├── rag_pipeline_huggingface_hosted.py  # HF hosted Inference API version, steps 1-6
│   └── rag_pipeline_pdf.py                 # PDF input version, steps 1-6
├── requirements.txt                         # covers every file above
├── .env                                     # your real API keys (gitignored, never commit this)
├── .env.example                             # template showing which variables to set
└── db/                                      # generated on every run, gitignored
    ├── qorebit/                               # from rag_pipeline.py
    ├── huggingface_local/                     # from rag_pipeline_huggingface.py
    ├── huggingface_hosted/                    # from rag_pipeline_huggingface_hosted.py
    ├── pdf/                                   # from rag_pipeline_pdf.py
    ├── chunking_<method>/                     # one per 01-chunking-methods/ file
    ├── loader_<method>/                       # one per 03-document-loaders/ file
    ├── memory_vectorstore/                    # from 04-memory/05_vectorstore_retriever_memory.py
    ├── chains_retrieval/                      # from 05-chains/09_retrieval_chain.py
    ├── chains_conversational/                 # from 05-chains/11_conversational_retrieval_chain.py
    ├── agents_retriever_tool/                 # from 06-agents/04_retriever_tool_agent.py
    ├── agents_multi_tool/                     # from 06-agents/05_multi_tool_agent.py
    ├── agents_retriever_tool_qorebit/          # from 06-agents/04_retriever_tool_agent_qorebit.py
    ├── agents_multi_tool_qorebit/              # from 06-agents/05_multi_tool_agent_qorebit.py
    ├── langgraph_multi_agent/                  # from 07-langgraph/05_multi_agent_graph.py
    ├── langgraph_multi_agent_qorebit/          # from 07-langgraph/05_multi_agent_graph_qorebit.py
    └── eval_rag/                                # from 08-evaluation/01_rag_eval.py
```

## Setup

### 1. Open a terminal in the project folder

Navigate to the project root — every command below assumes you're standing in this
directory (not inside `rag-app/` or `01-chunking-methods/`).

**PowerShell:**
```powershell
cd C:\Users\jimoh\OneDrive\Desktop\OFFICE\ai-engineering
```

**Git Bash:**
```bash
cd /c/Users/jimoh/OneDrive/Desktop/OFFICE/ai-engineering
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

### 5. Set up your API keys (only needed for a couple of files)

Most of this project needs no API key at all: every demo folder
(`01-chunking-methods/`, `02-prompt-engineering/`, `03-document-loaders/`, `04-memory/`,
`05-chains/`) and `rag_pipeline_huggingface.py` run on free local models, and so do the
non-`_qorebit.py` files in `06-agents/` and `07-langgraph/` (they use Ollama instead — see
[`06-agents/README.md`](06-agents/README.md) for its setup steps). Skip this step entirely
unless you plan to run
`rag_pipeline.py`, `rag_pipeline_pdf.py`, `rag_pipeline_huggingface_hosted.py`, or one of the
`_qorebit.py` files in `06-agents/` or `07-langgraph/`.

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
- `QOREBIT_API_KEY` — for `rag_pipeline.py`, `rag_pipeline_pdf.py`, the four
  `06-agents/*_qorebit.py` files, and the five `07-langgraph/*_qorebit.py` files. From your
  Qorebit dashboard → API Keys, looks like `qb_live_...`.
- `HUGGINGFACEHUB_API_TOKEN` — for `rag_pipeline_huggingface_hosted.py`. Get a free one at
  [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → "Create new
  token" → Read access is enough.

`.env` is listed in `.gitignore` — it should never be committed. `.env.example` holds
only placeholders and is safe to commit.

## Running it

Make sure your virtual environment is activated (prompt shows `(venv)`) and you're in the
project root. See [`rag-app/README.md`](rag-app/README.md) for the RAG pipeline's exact run
commands and setup per version, and [`01-chunking-methods/README.md`](01-chunking-methods/README.md)
for the chunking demos'.

The `08-evaluation/` folder runs the same way —
`python <folder>/<file>.py` from the project root, or `cd` into the folder first. It needs
no API key: its one LLM-backed file uses the free local `flan-t5-base` model. Ollama is the
one thing in this project that needs installing beyond `pip install` — see
[`06-agents/README.md`](06-agents/README.md) for the setup steps.

`09-deployment/01_fastapi_agent_service.py` is the one file that doesn't run once and exit —
`python 09-deployment/01_fastapi_agent_service.py` starts a server that keeps running until you
stop it (Ctrl+C), and you call it from another terminal instead (see its section above). It
needs Ollama, same as [`07-langgraph/`](07-langgraph/README.md).

Each step prints its own clearly-labeled section as it runs, so you can see exactly what's
happening — the chunks produced, what got stored, which passages matched your question, and
finally the generated answer.

The first run of any script downloads its models (a few hundred MB for embeddings, plus
~930MB more for `flan-t5-base` if you run the fully-local Hugging Face version) and caches
them locally in `~/.cache/huggingface` — later runs are fast, since nothing needs to be
re-downloaded.

## Trying your own questions

Open whichever file you're using and change the `QUESTION` constant near the top, then
rerun it. You can also swap in your own document by replacing `data/sample.txt` (or
changing `DOCUMENT_PATH`) and adjusting `CHUNK_SIZE` / `CHUNK_OVERLAP` if needed — repeat the
change in every file you want to use the same document, since each one sets these constants
independently.

## Notes / gotchas

- **"Failed to send telemetry event" warnings.** If you see these, it's a version
  mismatch between `chromadb` and a too-new `posthog` release (its `capture()` signature
  changed). `requirements.txt` pins `posthog<4` to avoid this — if it still shows up, run
  `pip install -r requirements.txt` again to make sure the pin took effect. It's harmless
  either way and never affects the pipeline's actual output.
- **Vector store resets on every run.** Every script deletes its own persisted Chroma
  folder under `db/` before rebuilding it, so re-running never duplicates chunks — it's not
  meant to persist across runs of a different document.
- **`langchain` and `langchain-community` are pinned newer than you might expect (0.3.30 /
  0.3.15, not the original 0.3.7).** `langchain-ollama` needs a `langchain-core` recent
  enough that its own `langsmith` floor no longer fits under the old `langchain==0.3.7`'s
  `langsmith<0.2.0` ceiling — a real, unavoidable resolver conflict, not a preference. Both
  bumps stay within the same 0.3.x line (no breaking changes), and the full test suite passed
  before and after the bump. If `pip install -r requirements.txt` ever reports a `langsmith`
  conflict again, it means one of these three packages' pins drifted apart — re-resolve them
  together rather than pinning `langsmith` directly.
