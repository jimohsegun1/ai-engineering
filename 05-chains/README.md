# Chain Composition

Part of the [AI Engineering From Scratch](../README.md) learning project — see the root
README for one-time setup (virtual environment, dependencies, API keys).

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

## Running it

```powershell
python 05-chains/01_simple_chain.py
python 05-chains/02_sequential_chain.py
python 05-chains/03_parallel_chain.py
python 05-chains/04_router_chain.py
python 05-chains/05_transform_chain.py
python 05-chains/06_stuff_documents_chain.py
python 05-chains/07_map_reduce_chain.py
python 05-chains/08_refine_chain.py
python 05-chains/09_retrieval_chain.py
python 05-chains/10_math_chain.py
python 05-chains/11_conversational_retrieval_chain.py
```
or `cd 05-chains` and run each file by its bare name. Either way, the venv needs to be
active and `data/`/`db/` are always resolved relative to the project root.

## Notes / gotchas

- **`flan-t5-base` is an unreliable model here.** It's the same small (~250M parameter)
  model used in `rag_pipeline_huggingface.py`, chosen for being free and CPU-friendly, not
  for quality. Expect `10_math_chain.py`'s expression parsing to fail more often than it
  succeeds — the file handles that gracefully rather than assuming success.
- **`10_math_chain.py` needs the `numexpr` package.** `LLMMathChain` evaluates the
  expression the LLM writes with `numexpr` rather than Python's own `eval`, so it depends on
  `numexpr` (in `requirements.txt`) in addition to `langchain`.

See the root [README](../README.md) for shared notes (telemetry warnings, vector store
reset behavior, dependency version pins) that apply across the whole project.
