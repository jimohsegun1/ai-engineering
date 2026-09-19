# Agents

Part of the [AI Engineering From Scratch](../README.md) learning project — see the root
README for one-time setup (virtual environment, dependencies, API keys).

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

## Ollama setup

Needed for the non-`_qorebit.py` files (and reused as-is by `07-langgraph/`):

1. Install [Ollama](https://ollama.com/download) (or `winget install Ollama.Ollama` on Windows) — it runs as a local background service.
2. Pull the model this folder uses: `ollama pull llama3.2:3b` (~2GB download, one-time).
3. That's it — no API key, and `langchain-ollama` is already in `requirements.txt`.

## Running it

```powershell
python 06-agents/01_react_agent.py                    # Ollama
python 06-agents/01_react_agent_qorebit.py             # Qorebit — needs QOREBIT_API_KEY in .env
python 06-agents/02_custom_tools.py                    # no LLM
python 06-agents/03_tool_calling_agent.py              # Ollama
python 06-agents/03_tool_calling_agent_qorebit.py      # Qorebit — needs QOREBIT_API_KEY in .env
python 06-agents/04_retriever_tool_agent.py            # Ollama
python 06-agents/04_retriever_tool_agent_qorebit.py    # Qorebit — needs QOREBIT_API_KEY in .env
python 06-agents/05_multi_tool_agent.py                # Ollama
python 06-agents/05_multi_tool_agent_qorebit.py        # Qorebit — needs QOREBIT_API_KEY in .env
```
or `cd 06-agents` and run each file by its bare name. Either way, the venv needs to be
active and `data/`/`db/` are always resolved relative to the project root.

## Notes / gotchas

- **The `_qorebit.py` files make live Qorebit calls, unlike the rest of this project's
  Qorebit usage.** Every other Qorebit-touching file comments out the actual API call by
  default so you can read/run the free steps and opt in to spending credits — but an agent
  demo has no free partial run to fall back to, the whole point is invoking the LLM, so
  these run live every time. They need `QOREBIT_API_KEY` in `.env`, same as `rag_pipeline.py`.
- **`01_react_agent.py` vs. `01_react_agent_qorebit.py` — a real before/after.** llama3.2:3b
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
- **`04_retriever_tool_agent.py` vs. its Qorebit twin** shows the same gap: llama3.2:3b
  sometimes calls the retriever tool for a question that doesn't need it, or even invents an
  unavailable tool name, instead of just answering directly — gpt-4o reliably calls the
  retriever only for the on-topic question and answers the other one directly with no tool
  call at all.
- **The three `_qorebit.py` files that use `create_tool_calling_agent` need
  `disable_streaming=True` on the LLM.** `AgentExecutor` streams the model's response
  internally to plan each step, and Qorebit's gateway mangles the `tool_call` id field when a
  tool-calling response is streamed — it comes back hundreds of characters long instead of a
  normal short id, and the next request then fails with a 400 ("string too long"). Disabling
  streaming makes the client request the full response in one piece instead, which avoids the
  bug entirely. `01_react_agent_qorebit.py` doesn't need this flag, since ReAct's plain-text
  format never involves a `tool_call` id in the first place.

See the root [README](../README.md) for shared notes (telemetry warnings, vector store
reset behavior, dependency version pins) that apply across the whole project.
