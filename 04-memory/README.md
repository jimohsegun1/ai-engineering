# Conversation Memory

Part of the [AI Engineering From Scratch](../README.md) learning project — see the root
README for one-time setup (virtual environment, dependencies, API keys).

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
`rag_pipeline_huggingface.py`, so they run free with no API key.

## Running it

```powershell
python 04-memory/01_buffer_memory.py
python 04-memory/02_buffer_window_memory.py
python 04-memory/03_summary_memory.py
python 04-memory/04_summary_buffer_memory.py
python 04-memory/05_vectorstore_retriever_memory.py
```
or `cd 04-memory` and run each file by its bare name. Either way, the venv needs to be
active and `data/`/`db/` are always resolved relative to the project root.

## Notes / gotchas

- **Classic memory classes are deprecated but still functional.** `langchain.memory` (used
  by all five files) prints a `LangChainDeprecationWarning` on import in LangChain 0.3.x —
  upstream now points toward LangGraph-based persistence instead (see
  `07-langgraph/04_persistent_memory.py`), but the classes still work fine for learning the
  underlying concepts.

See the root [README](../README.md) for shared notes (telemetry warnings, vector store
reset behavior, dependency version pins) that apply across the whole project.
