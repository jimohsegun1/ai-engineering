# Observability & Tracing

Part of the [AI Engineering From Scratch](../README.md) learning project — see the root
README for one-time setup (virtual environment, dependencies, API keys).

Deploying LLMs and RAG pipelines to production requires answering crucial diagnostic questions:
- *Why is a specific user request taking 4.2 seconds? (Is it embedding, vector search, or token generation?)*
- *What exact prompt with augmented chunks was dispatched to the LLM?*
- *How many prompt and completion tokens were spent, and what did that call cost?*
- *Which tool or retriever step threw an exception or returned empty context?*

This module explores the three primary observability paradigms in AI Engineering:

1. **Custom Callbacks (`BaseCallbackHandler`)**:
   Zero-cloud, local profiling tracker hooking into LangChain's event bus. Computes millisecond latencies for each pipeline phase and renders an ASCII flame-graph breakdown directly in your terminal.
2. **OpenTelemetry (OTEL)**:
   Vendor-neutral, open standard distributed tracing instrumented with GenAI Semantic Conventions (`gen_ai.system`, `gen_ai.request.model`, `rag.retrieval`, `db.system.name`). Emits structured JSON spans exportable to Jaeger, Datadog, Grafana Tempo, or Honeycomb.
3. **LangSmith Tracing**:
   LangChain's production telemetry platform capturing hierarchical run trees, input/output diffs, token accounting, and custom user/tenant metadata tags. Includes both local HuggingFace and hosted Qorebit (GPT-4o) implementations with automatic dry-run fallback.

---

## Observability Architecture

```mermaid
flowchart TD
    subgraph Client ["Client Layer"]
        Q["User Query"]
    end

    subgraph Tracer ["Observability & Telemetry Instrumentation"]
        direction TB
        CB["Custom BaseCallbackHandler<br/>(Terminal Flame-Graph)"]
        OTEL["OpenTelemetry Tracer<br/>(GenAI Semantic Spans)"]
        LS["LangSmith Cloud / Local Runner<br/>(Hierarchical Run Trees)"]
    end

    subgraph Pipeline ["Traced RAG Pipeline"]
        direction TB
        SPLIT["Document Splitter & Indexer"]
        RET["Chroma Vector Retriever<br/>(run_name: 'KnowledgeBaseRetriever')"]
        PRMPT["Prompt Template Formatter<br/>(run_name: 'RagPromptFormatter')"]
        GEN["LLM Text Generation<br/>(run_name: 'AnswerGeneratorLLM')"]
        PARSE["StrOutputParser<br/>(run_name: 'OutputParser')"]
    end

    Q --> CB
    Q --> OTEL
    Q --> LS

    CB -.->|on_chain_start| RET
    RET -->|on_retriever_end (latency)| PRMPT
    PRMPT -->|on_llm_start| GEN
    GEN -->|on_llm_end (token metrics)| PARSE
    PARSE -->|on_chain_end| OUT["Final Traced Answer"]
```

---

## Files in this module

| File | Paradigm | Models Used | Key Telemetry Capabilities |
| --- | --- | --- | --- |
| `01_custom_callbacks_tracing.py` | Native Callbacks | `all-MiniLM-L6-v2` + `flan-t5-base` | In-memory latency profiling, step timing, token metrics, terminal ASCII flame-graph. |
| `02_opentelemetry_tracing.py` | OpenTelemetry | `all-MiniLM-L6-v2` + `flan-t5-base` | OTEL TracerProvider, GenAI semantic spans, parent-child span hierarchy, JSON exporter. |
| `03_langsmith_tracing.py` | LangSmith | `all-MiniLM-L6-v2` + `flan-t5-base` | Hierarchical run names, environment configuration, run tags, metadata dictionaries, dry-run mode. |
| `03_langsmith_tracing_qorebit.py` | LangSmith (Hosted) | `all-MiniLM-L6-v2` + `openai/gpt-4o` | Commercial model tracing, token accounting, TTFT latency, production user metadata. |

---

## Observability Comparison Matrix

| Dimension | Custom Callbacks | OpenTelemetry (OTEL) | LangSmith Tracing |
| --- | --- | --- | --- |
| **Setup Complexity** | Lowest (Standard Library + LangChain) | Moderate (SDK + TracerProvider) | Lowest (Environment Variables) |
| **Vendor Neutrality** | High (Pure Python) | Highest (Industry W3C Standard) | Low (LangChain Ecosystem) |
| **External Cloud Required** | None (100% offline) | Optional (Console / OTLP Collector) | Optional (Works offline / dry-run) |
| **GenAI Semantic Conventions** | Manual attributes | Native (`gen_ai.*` standard attributes) | Proprietary LangSmith Schema |
| **Flame-Graph / Tree View** | Terminal ASCII flame-graph | APM UIs (Tempo, Jaeger, Datadog) | Rich Web UI with input/output diffs |
| **Token & Cost Accounting** | Extracted from `LLMResult` | Captured as span attributes | Automatic pricing tiers & cost models |

---

## Running the Examples

Ensure your virtual environment is activated from the repository root:

```powershell
venv\Scripts\Activate.ps1
```

### 1. Custom Callbacks & Flame-Graph
Runs completely locally with zero API keys or external services:

```powershell
python 15-observability/01_custom_callbacks_tracing.py
```

*Example Terminal Output:*
```text
======================================================================
OBSERVABILITY REPORT: COMPONENT LATENCIES & METRICS
======================================================================
Total Pipeline Duration: 1.48s
Tokens Tracked:          28 prompt + 24 completion = 52 total

COMPONENT BREAKDOWN:
  [========================================] 82.4% ( 1.22s) LLM Generation
  [========                                ] 16.8% ( 0.25s) Vector Retrieval
  [=                                       ]  0.8% ( 0.01s) Prompt & Chain Plumbing
======================================================================
```

### 2. OpenTelemetry Distributed Tracing
Instruments RAG retrieval and LLM generation using OpenTelemetry spans and emits structured JSON span logs:

```powershell
python 15-observability/02_opentelemetry_tracing.py
```

### 3. LangSmith Tracing (Local HuggingFace)
Inspects LangSmith configuration, tags metadata, assigns human-readable `run_name` labels, and runs the chain:

```powershell
python 15-observability/03_langsmith_tracing.py
```

### 4. LangSmith Tracing with Qorebit (GPT-4o)
Executes a traced commercial GPT-4o pipeline capturing full token analytics:

```powershell
python 15-observability/03_langsmith_tracing_qorebit.py
```

---

## Key Takeaways

1. **Name Every Node with `run_name`**:
   Default LangChain traces can be hard to decipher when multiple chains or runnables are nested. Adding `.with_config(run_name="CustomDescriptiveName")` produces clear, readable trace trees.
2. **Never Let Observability Block Application Execution**:
   Production tracing should be asynchronous or non-blocking. Both LangSmith and OpenTelemetry buffer spans in background worker threads so network hiccups to telemetry backends never fail user requests.
3. **Attach High-Cardinality Metadata**:
   Always attach `user_id`, `tenant_id`, `environment`, and `model_version` to your trace configs. When debugging an outage or quality regression, filtering traces by metadata isolates problems in seconds.
