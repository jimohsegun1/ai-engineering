# Deployment

Part of the [AI Engineering From Scratch](../README.md) learning project — see the root
README for one-time setup (virtual environment, dependencies, API keys).

Every earlier demo calls `graph.invoke(...)` or `chain.invoke(...)` once, in-process, then
exits. `01_fastapi_agent_service.py` takes the supervisor graph from
[`07-langgraph/07_supervisor_agent.py`](../07-langgraph/README.md) and wraps it in a small
[FastAPI](https://fastapi.tiangolo.com/) app instead, so it runs as a long-lived HTTP service
other programs can call:

- `GET /health` — a plain liveness check
- `POST /chat` — `{"question": "..."}` in, `{"route": "...", "answer": "..."}` out

The graph (and its connection to Ollama) is built once at server startup rather than once per
request. Needs the same Ollama setup as [`07-langgraph/`](../07-langgraph/README.md) — no API
key.

## Running it

`01_fastapi_agent_service.py` is the one file in this project that doesn't run once and
exit — it starts a server that keeps running until you stop it (Ctrl+C), and you call it
from another terminal instead:

```powershell
python 09-deployment/01_fastapi_agent_service.py
```

```powershell
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\": \"What is 24 times 7, plus 10?\"}"
```

See the root [README](../README.md) for shared notes (telemetry warnings, vector store
reset behavior, dependency version pins) that apply across the whole project.
