# AI Engineering From Scratch

A collection of small, standalone LangChain demos written as a learning project — a RAG
pipeline plus focused demo folders for document loaders, chunking methods, conversation
memory, chain composition, and prompting techniques. Every file runs on its own; none of the
demo folders depend on each other.

## RAG pipeline

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
directory, so you can run them either from the project root:
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
relative to the project root, never relative to `chunking_methods/`.

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
python rag-app/rag_pipeline_pdf.py
```

### Document loader demos

`document-loader/` has six standalone files, each demonstrating a different LangChain
document loader (step 1) — everything else about the pipeline (or, for the smaller formats,
what's left of it) stays the same. Step 6 (the Qorebit call) is commented out in every file,
same as `chunking_methods/`:

| File | Loader | Input | What it shows |
| --- | --- | --- | --- |
| `01_text_loader.py` | `TextLoader` | `data/sample.txt` | The baseline: one Document for a whole plain-text file |
| `02_pdf_loader.py` | `PyPDFLoader` | `data/sample.pdf` | One Document per PDF page, with a `page` number in metadata |
| `03_csv_loader.py` | `CSVLoader` | `data/sample.csv` | One Document per row, formatted as `column: value` lines |
| `04_json_loader.py` | `JSONLoader` | `data/sample.json` | Pulls one Document per array element out of nested JSON with a jq schema |
| `05_directory_loader.py` | `DirectoryLoader` | `data/sample.txt` + `data/sample2.txt` | Loads every file matching a glob pattern in one call, instead of naming files one by one |
| `06_web_loader.py` | `WebBaseLoader` | a live web page | The only loader here that needs internet access, parsed with BeautifulSoup |

For the row/record-shaped formats (CSV, JSON) chunking is skipped entirely — each row or
record is already a small, self-contained unit of text, so step 2 just passes the documents
through as-is. Run these the same way as `chunking_methods/`, either from the project root
or from inside `document-loader/` itself.

## Memory demos

`memory/` has five standalone files, each demonstrating a different LangChain conversation
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

`chains/` has five standalone files, each demonstrating a different way to compose LCEL
(`|`-piped) chains together. All five run on the local, free `google/flan-t5-base` model, so
there's no Qorebit step to comment out here:

| File | Pattern | What it shows |
| --- | --- | --- |
| `01_simple_chain.py` | `prompt \| llm \| output_parser` | The basic three-piece chain shape every other file builds on |
| `02_sequential_chain.py` | `RunnablePassthrough.assign` | Feeds one chain's output into a second chain's input, keeping every intermediate value |
| `03_parallel_chain.py` | `RunnableParallel` | Runs two independent chains against the same input at once instead of one after another |
| `04_router_chain.py` | `RunnableBranch` | Sends an input down one of several chains depending on a condition |
| `05_transform_chain.py` | `RunnableLambda` | A plain-Python transform step inserted into a chain — not every step has to call an LLM |

## Prompt engineering demos

`prompt-engineering/` has five standalone files, each demonstrating a different prompting
technique on the same local `google/flan-t5-base` model:

| File | Technique | What it shows |
| --- | --- | --- |
| `01_zero_shot_prompting.py` | Zero-shot | Just an instruction, no examples of the desired output |
| `02_few_shot_prompting.py` | Few-shot (`FewShotPromptTemplate`) | A handful of labeled examples before the real question — compare with the file above |
| `03_chain_of_thought_prompting.py` | Chain-of-thought | Asking the model to reason step by step, compared against a direct-answer prompt on the same question |
| `04_role_based_prompting.py` | Role-based | The same question answered twice, once per assigned persona |
| `05_structured_output_prompting.py` | Structured output (`PydanticOutputParser`) | Asks for JSON matching a schema, and handles the (likely) case where a small model doesn't follow it |

`flan-t5-base` is small enough to be an unreliable narrator for some of these — chain-of-thought
answers can loop or drift, and structured-output parsing often fails outright. That's noted in
each file and is expected; the point is to see the prompting mechanics work, not to get
perfect answers out of a ~250M parameter model.

## Stack

- **Document loading / chunking**: LangChain (`TextLoader` or `PyPDFLoader`, `RecursiveCharacterTextSplitter`)
- **Embeddings**: local, free HuggingFace model (`sentence-transformers/all-MiniLM-L6-v2`) — no API key or cost
- **Vector database**: Chroma — every file persists under `db/`, each to its own
  subfolder (e.g. `db/qorebit/`, `db/huggingface_local/`, `db/chunking_code/`) so running
  one never clobbers another's store
- **LLM (generation)**: Qorebit, a local `flan-t5-base` model, or Hugging Face's hosted
  Inference API — see the table above for which file uses which

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
├── chunking_methods/                     # six chunking-method demos, see table above
│   ├── 01_character_splitter.py
│   ├── 02_recursive_character_splitter.py
│   ├── 03_token_splitter.py
│   ├── 04_markdown_header_splitter.py
│   ├── 05_semantic_chunker.py
│   └── 06_code_splitter.py
├── document-loader/                      # six document-loader demos, see table above
│   ├── 01_text_loader.py
│   ├── 02_pdf_loader.py
│   ├── 03_csv_loader.py
│   ├── 04_json_loader.py
│   ├── 05_directory_loader.py
│   └── 06_web_loader.py
├── memory/                                # five conversation-memory demos, see table above
│   ├── 01_buffer_memory.py
│   ├── 02_buffer_window_memory.py
│   ├── 03_summary_memory.py
│   ├── 04_summary_buffer_memory.py
│   └── 05_vectorstore_retriever_memory.py
├── chains/                                # five LCEL chain-composition demos, see table above
│   ├── 01_simple_chain.py
│   ├── 02_sequential_chain.py
│   ├── 03_parallel_chain.py
│   ├── 04_router_chain.py
│   └── 05_transform_chain.py
├── prompt-engineering/                    # five prompting-technique demos, see table above
│   ├── 01_zero_shot_prompting.py
│   ├── 02_few_shot_prompting.py
│   ├── 03_chain_of_thought_prompting.py
│   ├── 04_role_based_prompting.py
│   └── 05_structured_output_prompting.py
├── rag-app/
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
    ├── chunking_<method>/                     # one per chunking_methods/ file
    ├── loader_<method>/                       # one per document-loader/ file
    └── memory_vectorstore/                    # from memory/05_vectorstore_retriever_memory.py
```

## Setup

### 1. Open a terminal in the project folder

Navigate to the project root — every command below assumes you're standing in this
directory (not inside `rag-app/` or `chunking_methods/`).

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
project root, then run whichever version you want:

```powershell
python rag-app/rag_pipeline.py                          # Qorebit — needs QOREBIT_API_KEY in .env
python rag-app/rag_pipeline_huggingface.py              # fully local — no setup needed beyond step 4
python rag-app/rag_pipeline_huggingface_hosted.py       # HF hosted API — needs HUGGINGFACEHUB_API_TOKEN in .env
python rag-app/rag_pipeline_pdf.py                      # PDF input — needs QOREBIT_API_KEY in .env
```

The demo folders (`chunking_methods/`, `document-loader/`, `memory/`, `chains/`,
`prompt-engineering/`) run the same way — `python <folder>/<file>.py` from the project root,
or `cd` into the folder first. None of them need an API key: the RAG-style ones use Qorebit
only for the commented-out step 6, and everything in `memory/`, `chains/`, and
`prompt-engineering/` that needs an LLM at all uses the free local `flan-t5-base` model.

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
- **`document-loader/04_json_loader.py` needs the `jq` package.** `JSONLoader` uses jq
  schemas to pull data out of nested JSON, so it depends on the `jq` Python bindings
  (in `requirements.txt`) rather than just `langchain-community`.
- **`document-loader/06_web_loader.py` needs internet access and `beautifulsoup4`.** It's
  the only loader demo that calls out to a live URL instead of reading a local file. It also
  sets a `USER_AGENT` environment variable to avoid a harmless warning some sites' servers
  trigger when it's unset.
- **`memory/`'s classic memory classes are deprecated but still functional.**
  `langchain.memory` (used by all five files) prints a `LangChainDeprecationWarning` on
  import in LangChain 0.3.x — upstream now points toward LangGraph-based persistence
  instead, but the classes still work fine for learning the underlying concepts.
- **`flan-t5-base` is an unreliable model for `chains/` and `prompt-engineering/`.** It's
  the same small (~250M parameter) model used in `rag_pipeline_huggingface.py`, chosen for
  being free and CPU-friendly, not for quality. Expect chain-of-thought answers to sometimes
  loop or drift, and expect `05_structured_output_prompting.py`'s JSON parsing to fail more
  often than it succeeds — both files handle that gracefully rather than assuming success.
