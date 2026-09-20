# LangGraph

Part of the [AI Engineering From Scratch](../README.md) learning project — see the root
README for one-time setup (virtual environment, dependencies, API keys).

[`06-agents/`](../06-agents/README.md) builds agents by hand with the older `AgentExecutor`.
LangGraph is the newer, more general framework underneath: instead of a fixed agent loop, you
define a graph of nodes that read and write a shared state, and LangGraph handles running it.
`07-langgraph/` has twelve standalone files: seven run on [Ollama](https://ollama.com)
(`llama3.2:3b`, local and free), and five of those seven (every file with an LLM in it) have
a Qorebit-backed twin (`_qorebit.py` suffix) using `gpt-4o`, the same comparison
`06-agents/` sets up:

| File | Concept | What it shows |
| --- | --- | --- |
| `01_simple_graph.py` | `StateGraph`, nodes, edges | The fundamental mechanic — two plain-function nodes run in sequence, no LLM at all, no Qorebit twin needed |
| `02_conditional_graph.py` | `add_conditional_edges` | Routes to one of two nodes based on a rule — the graph-based equivalent of `05-chains/04_router_chain.py` — no LLM, no Qorebit twin needed |
| `03_tool_calling_agent.py` / `_qorebit.py` | `langgraph.prebuilt.create_react_agent` | The same tool-calling agent as `06-agents/03_tool_calling_agent.py`, built in one call instead of assembling a prompt + executor by hand |
| `04_persistent_memory.py` / `_qorebit.py` | `MemorySaver` checkpointer + `thread_id` | The agent remembers earlier turns automatically — the modern replacement for wrapping `04-memory/`'s memory classes around an agent |
| `05_multi_agent_graph.py` / `_qorebit.py` | multiple specialized nodes | A retriever-only "researcher" node feeds an LLM-backed "writer" node — a basic multi-node composition, one step short of a full multi-agent supervisor |
| `06_streaming.py` / `_qorebit.py` | `graph.stream()` | Two stream modes side by side: `"updates"` (one event per finished node) and `"messages"` (LLM tokens as they're generated, across every node) |
| `07_supervisor_agent.py` / `_qorebit.py` | supervisor multi-agent pattern | An LLM-based supervisor node classifies each question and routes it to one of three specialist workers (math, writing, general) via `add_conditional_edges` — the general version of `02_conditional_graph.py`'s hand-written rule |

Needs the same Ollama setup as [`06-agents/`](../06-agents/README.md) — no API key.

## Running it

```powershell
python 07-langgraph/01_simple_graph.py
python 07-langgraph/02_conditional_graph.py
python 07-langgraph/03_tool_calling_agent.py              # Ollama
python 07-langgraph/03_tool_calling_agent_qorebit.py      # Qorebit — needs QOREBIT_API_KEY in .env
python 07-langgraph/04_persistent_memory.py               # Ollama
python 07-langgraph/04_persistent_memory_qorebit.py       # Qorebit — needs QOREBIT_API_KEY in .env
python 07-langgraph/05_multi_agent_graph.py               # Ollama
python 07-langgraph/05_multi_agent_graph_qorebit.py       # Qorebit — needs QOREBIT_API_KEY in .env
python 07-langgraph/06_streaming.py                       # Ollama
python 07-langgraph/06_streaming_qorebit.py               # Qorebit — needs QOREBIT_API_KEY in .env
python 07-langgraph/07_supervisor_agent.py                # Ollama
python 07-langgraph/07_supervisor_agent_qorebit.py        # Qorebit — needs QOREBIT_API_KEY in .env
```
or `cd 07-langgraph` and run each file by its bare name. Either way, the venv needs to be
active and `data/`/`db/` are always resolved relative to the project root.

## Notes / gotchas

- **Harmless deprecation warning on every run.** You'll see a
  `LangChainPendingDeprecationWarning` about `allowed_objects`; it comes from LangGraph's own
  checkpoint-serialization internals, not from anything in these files, and doesn't affect
  the output.
- **The `_qorebit.py` files make live Qorebit calls**, same as `06-agents/`'s twins — no free
  partial run to fall back to, so they run live every time and need `QOREBIT_API_KEY` in
  `.env`. Only `03_tool_calling_agent_qorebit.py` needs `disable_streaming=True`, for the
  same `tool_call`-id-mangling reason documented in
  [`06-agents/README.md`](../06-agents/README.md) — the other four Qorebit twins either bind
  no tools or (in `06_streaming_qorebit.py`'s case) specifically need streaming left on to
  demonstrate `stream_mode="messages"`.

See the root [README](../README.md) for shared notes (telemetry warnings, vector store
reset behavior, dependency version pins) that apply across the whole project.
