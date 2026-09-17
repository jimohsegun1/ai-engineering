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

## Memory demos

`04-memory/` has five standalone files, each demonstrating a different LangChain conversation
memory type. Unlike the other demo folders, these don't run a retrieval pipeline — they save
a fixed script of conversation turns into memory and print what each type retains after
every turn, so you can compare them directly:

| File | Memory type | Needs an LLM? | What it shows |
| --- | --- | --- | --- |
| `01_buffer_memory.py` | `ConversationBufferMemory` | No | Keeps the full transcript verbatim — keeps growing forever |
| `02_buffer_window_memory.py` | `ConversationBufferWindowMemory` | No | Keeps only the last `k` exchanges, drops everything older |
| `03_summary_memory.py` | `ConversationSummaryMemory` | Yes (local flan-t5) | Rewrites the whole transcript into a running summary after every turn |
| `04_summary_buffer_memory.py` | `ConversationSummaryBufferMemory` | Yes (local flan-t5) | Recent turns kept verbatim, older ones rolled into a summary once a token limit is hit |
| `05_vectorstore_retriever_memory.py` | `VectorStoreRetrieverMemory` | No | Retrieves whichever *past* exchange is semantically closest to the new input, via Chroma |

The two summary-based files use the same local `google/flan-t5-base` model as
`rag_pipeline_huggingface.py`, so they run free with no API key. You'll see a
`LangChainDeprecationWarning` when importing `langchain.memory` — these classes still work
in LangChain 0.3.x, but upstream now recommends LangGraph-based persistence instead; harmless
for this learning project.

## Chain composition demos

`05-chains/` has eleven standalone files, each demonstrating a different way to compose
chains together — five general LCEL composition patterns, plus six of LangChain's dedicated
document/retrieval/utility chain constructors. All eleven run on the local, free
`google/flan-t5-base` model, so there's no Qorebit step to comment out here:

| File | Pattern | What it shows |
| --- | --- | --- |
| `01_simple_chain.py` | `prompt \| llm \| output_parser` | The basic three-piece chain shape every other file builds on |
| `02_sequential_chain.py` | `RunnablePassthrough.assign` | Feeds one chain's output into a second chain's input, keeping every intermediate value |
| `03_parallel_chain.py` | `RunnableParallel` | Runs two independent chains against the same input at once instead of one after another |
| `04_router_chain.py` | `RunnableBranch` | Sends an input down one of several chains depending on a condition |
| `05_transform_chain.py` | `RunnableLambda` | A plain-Python transform step inserted into a chain — not every step has to call an LLM |
| `06_stuff_documents_chain.py` | `create_stuff_documents_chain` | The official building block for the "concatenate every doc into one prompt" pattern rag-app builds by hand |
| `07_map_reduce_chain.py` | `load_summarize_chain(chain_type="map_reduce")` | Summarizes each document independently, then combines those summaries into one |
| `08_refine_chain.py` | `load_summarize_chain(chain_type="refine")` | Processes documents one at a time, refining a running summary instead of combining independent ones |
| `09_retrieval_chain.py` | `create_retrieval_chain` | The modern replacement for the legacy `RetrievalQA` chain — retriever + stuff-documents chain in one call |
| `10_math_chain.py` | `LLMMathChain` | The LLM writes a Python expression for a word problem, then `numexpr` evaluates it instead of trusting the model's arithmetic |
| `11_conversational_retrieval_chain.py` | `create_history_aware_retriever` + `create_retrieval_chain` | A RAG chain with memory — rewrites a vague follow-up ("why is it useful?") into a standalone question using chat history *before* retrieving; the modern replacement for the legacy `ConversationalRetrievalChain` |

## Agent demos

Unlike a chain, an agent doesn't follow a fixed sequence — the LLM decides at each turn
whether to call a tool or give a final answer, based on its own reasoning. `06-agents/` has
nine standalone files: five run on [Ollama](https://ollama.com) (`llama3.2:3b`, local and
free) and four of those five have a Qorebit-backed twin (`_qorebit.py` suffix) using
`gpt-4o`, so you can directly compare a small local model against a larger hosted one on the
exact same task. This is the one folder that **doesn't** use `flan-t5-base` — reliable tool
use needs a real instruction-tuned model:

| File | Pattern | What it shows |
| --- | --- | --- |
| `01_react_agent.py` / `_qorebit.py` | `create_react_agent` (ReAct) | The classic text-based Thought/Action/Observation loop — brittle on llama3.2:3b, works (after retries) on gpt-4o, see the notes below |
| `02_custom_tools.py` | `@tool` | How a plain Python function becomes something an agent can be told about and call — no LLM involved, no Qorebit twin needed |
| `03_tool_calling_agent.py` / `_qorebit.py` | `create_tool_calling_agent` | The modern replacement for ReAct: the model returns a structured tool call directly instead of text to parse |
| `04_retriever_tool_agent.py` / `_qorebit.py` | `create_retriever_tool` | Wraps a Chroma retriever as a tool so the agent decides for itself whether a question needs a document lookup |
| `05_multi_tool_agent.py` / `_qorebit.py` | multiple tools on one agent | A calculator, a word counter, and a retriever together — asks several questions to see which tool (if any) gets picked each time |

**Ollama setup** (only needed for the non-`_qorebit.py` files):
1. Install [Ollama](https://ollama.com/download) (or `winget install Ollama.Ollama` on Windows) — it runs as a local background service.
2. Pull the model this folder uses: `ollama pull llama3.2:3b` (~2GB download, one-time).
3. That's it — no API key, and `langchain-ollama` is already in `requirements.txt`.

**The `_qorebit.py` files make live Qorebit calls, unlike the rest of this project's Qorebit
usage.** Every other Qorebit-touching file comments out the actual API call by default so you
can read/run the free steps and opt in to spending credits — but an agent demo has no free
partial run to fall back to, the whole point is invoking the LLM, so these run live every
time. They need `QOREBIT_API_KEY` in `.env`, same as `rag_pipeline.py`.

**`01_react_agent.py` vs. `01_react_agent_qorebit.py` — a real before/after.** llama3.2:3b
often computes the correct answer via the calculator tool — repeatedly — but never actually
writes the `Final Answer:` line the ReAct parser is watching for, so it loops until
`max_iterations` (set to 5) cuts it off and returns "Agent stopped due to iteration limit."
gpt-4o via Qorebit does reach the correct final answer, but not cleanly on the first try
either: it tends to write out the whole Thought/Action/Observation/Final Answer sequence in
one go, predicting the tool's result instead of waiting for it — `handle_parsing_errors=True`
catches this and retries until the model settles into the correct format. Both are real,
reproducible behaviors, not bugs in the code; that's exactly why `03_tool_calling_agent.py`
exists — the same kind of task works cleanly on both models once native tool-calling is used
instead of text parsing.

**`04_retriever_tool_agent.py` vs. its Qorebit twin** shows the same gap: llama3.2:3b
sometimes calls the retriever tool for a question that doesn't need it, or even invents an
unavailable tool name, instead of just answering directly — gpt-4o reliably calls the
retriever only for the on-topic question and answers the other one directly with no tool call
at all.

**The three `_qorebit.py` files that use `create_tool_calling_agent` need
`disable_streaming=True` on the LLM.** `AgentExecutor` streams the model's response
internally to plan each step, and Qorebit's gateway mangles the `tool_call` id field when a
tool-calling response is streamed — it comes back hundreds of characters long instead of a
normal short id, and the next request then fails with a 400 ("string too long"). Disabling
streaming makes the client request the full response in one piece instead, which avoids the
bug entirely. `01_react_agent_qorebit.py` doesn't need this flag, since ReAct's plain-text
format never involves a `tool_call` id in the first place.

## LangGraph demos

`06-agents/` builds agents by hand with the older `AgentExecutor`. LangGraph is the newer,
more general framework underneath: instead of a fixed agent loop, you define a graph of nodes
that read and write a shared state, and LangGraph handles running it. `07-langgraph/` has
twelve standalone files: seven run on [Ollama](https://ollama.com) (`llama3.2:3b`, local and
free), and five of those seven (every file with an LLM in it) have a Qorebit-backed twin
(`_qorebit.py` suffix) using `gpt-4o`, the same comparison `06-agents/` sets up:

| File | Concept | What it shows |
| --- | --- | --- |
| `01_simple_graph.py` | `StateGraph`, nodes, edges | The fundamental mechanic — two plain-function nodes run in sequence, no LLM at all, no Qorebit twin needed |
| `02_conditional_graph.py` | `add_conditional_edges` | Routes to one of two nodes based on a rule — the graph-based equivalent of `05-chains/04_router_chain.py` — no LLM, no Qorebit twin needed |
| `03_tool_calling_agent.py` / `_qorebit.py` | `langgraph.prebuilt.create_react_agent` | The same tool-calling agent as `06-agents/03_tool_calling_agent.py`, built in one call instead of assembling a prompt + executor by hand |
| `04_persistent_memory.py` / `_qorebit.py` | `MemorySaver` checkpointer + `thread_id` | The agent remembers earlier turns automatically — the modern replacement for wrapping `04-memory/`'s memory classes around an agent |
| `05_multi_agent_graph.py` / `_qorebit.py` | multiple specialized nodes | A retriever-only "researcher" node feeds an LLM-backed "writer" node — a basic multi-node composition, one step short of a full multi-agent supervisor |
| `06_streaming.py` / `_qorebit.py` | `graph.stream()` | Two stream modes side by side: `"updates"` (one event per finished node) and `"messages"` (LLM tokens as they're generated, across every node) |
| `07_supervisor_agent.py` / `_qorebit.py` | supervisor multi-agent pattern | An LLM-based supervisor node classifies each question and routes it to one of three specialist workers (math, writing, general) via `add_conditional_edges` — the general version of `02_conditional_graph.py`'s hand-written rule |

Needs the same Ollama setup as `06-agents/` (see its section above) — no API key. You'll see a
harmless `LangChainPendingDeprecationWarning` about `allowed_objects` on every run; it comes
from LangGraph's own checkpoint-serialization internals, not from anything in these files, and
doesn't affect the output.

**The `_qorebit.py` files make live Qorebit calls**, same as `06-agents/`'s twins — no free
partial run to fall back to, so they run live every time and need `QOREBIT_API_KEY` in `.env`.
Only `03_tool_calling_agent_qorebit.py` needs `disable_streaming=True`, for the same
`tool_call`-id-mangling reason documented in the Agent demos section above — the other four
Qorebit twins either bind no tools or (in `06_streaming_qorebit.py`'s case) specifically need
streaming left on to demonstrate `stream_mode="messages"`.

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
request. Needs the same Ollama setup as `07-langgraph/` — no API key.

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
├── 04-memory/                             # five conversation-memory demos, see table above
│   ├── 01_buffer_memory.py
│   ├── 02_buffer_window_memory.py
│   ├── 03_summary_memory.py
│   ├── 04_summary_buffer_memory.py
│   └── 05_vectorstore_retriever_memory.py
├── 05-chains/                             # eleven chain-composition demos, see table above
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
├── 06-agents/                             # nine agent demos, see table above
│   ├── 01_react_agent.py                     # Ollama
│   ├── 01_react_agent_qorebit.py             # Qorebit
│   ├── 02_custom_tools.py                    # no LLM
│   ├── 03_tool_calling_agent.py               # Ollama
│   ├── 03_tool_calling_agent_qorebit.py       # Qorebit
│   ├── 04_retriever_tool_agent.py             # Ollama
│   ├── 04_retriever_tool_agent_qorebit.py     # Qorebit
│   ├── 05_multi_tool_agent.py                 # Ollama
│   └── 05_multi_tool_agent_qorebit.py         # Qorebit
├── 07-langgraph/                          # twelve LangGraph demos, see table above (needs Ollama)
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
its own setup steps in that section). Skip this step entirely unless you plan to run
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

The demo folders (`04-memory/`, `05-chains/`,
`06-agents/`, `07-langgraph/`, `08-evaluation/`) run the same way —
`python <folder>/<file>.py` from the project root, or `cd` into the folder first. Almost none
of them need an API key: the RAG-style ones use Qorebit only for the commented-out step 6,
everything in `04-memory/`, `05-chains/`, and `08-evaluation/` that
needs an LLM at all uses the free local `flan-t5-base` model, and most files in `06-agents/`
and `07-langgraph/` use the free local Ollama model instead (see the setup steps above —
Ollama is the one thing in this project that needs installing beyond `pip install`). The
exception is each folder's `*_qorebit.py` files (four in `06-agents/`, five in
`07-langgraph/`), which need `QOREBIT_API_KEY` and make a real, live Qorebit call every time
you run them.

`09-deployment/01_fastapi_agent_service.py` is the one file that doesn't run once and exit —
`python 09-deployment/01_fastapi_agent_service.py` starts a server that keeps running until you
stop it (Ctrl+C), and you call it from another terminal instead (see its section above). It
needs Ollama, same as `07-langgraph/`.

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
- **`04-memory/`'s classic memory classes are deprecated but still functional.**
  `langchain.memory` (used by all five files) prints a `LangChainDeprecationWarning` on
  import in LangChain 0.3.x — upstream now points toward LangGraph-based persistence
  instead, but the classes still work fine for learning the underlying concepts.
- **`flan-t5-base` is an unreliable model for `05-chains/`.** It's the same small (~250M
  parameter) model used in `rag_pipeline_huggingface.py`, chosen for being free and
  CPU-friendly, not for quality. Expect `05-chains/10_math_chain.py`'s expression parsing to
  fail more often than it succeeds — the file handles that gracefully rather than assuming
  success.
- **`05-chains/10_math_chain.py` needs the `numexpr` package.** `LLMMathChain` evaluates the
  expression the LLM writes with `numexpr` rather than Python's own `eval`, so it depends on
  `numexpr` (in `requirements.txt`) in addition to `langchain`.
- **`langchain` and `langchain-community` are pinned newer than you might expect (0.3.30 /
  0.3.15, not the original 0.3.7).** `langchain-ollama` needs a `langchain-core` recent
  enough that its own `langsmith` floor no longer fits under the old `langchain==0.3.7`'s
  `langsmith<0.2.0` ceiling — a real, unavoidable resolver conflict, not a preference. Both
  bumps stay within the same 0.3.x line (no breaking changes), and the full test suite passed
  before and after the bump. If `pip install -r requirements.txt` ever reports a `langsmith`
  conflict again, it means one of these three packages' pins drifted apart — re-resolve them
  together rather than pinning `langsmith` directly.
