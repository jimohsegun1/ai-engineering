"""Observability: Standardized OpenTelemetry (OTEL) Distributed Tracing.

OpenTelemetry is the CNCF standard for vendor-neutral observability.
While LangChain-specific dashboards (like LangSmith) are powerful, enterprise
environments often require telemetry to flow into enterprise monitoring platforms
(Datadog, Dynatrace, New Relic, Honeycomb, Jaeger, Grafana Tempo).

This file demonstrates instrumenting an AI pipeline using OpenTelemetry:
  1. Initializing a TracerProvider with a ConsoleSpanExporter.
  2. Creating hierarchical Spans:
     - Root Span: `rag.query_pipeline`
     - Child Span 1: `chroma.similarity_search` (with DB semantic attributes)
     - Child Span 2: `gen_ai.generation` (with GenAI semantic attributes)
  3. Recording telemetry events, exceptions, and latency directly to the span.

Runs 100% locally with zero external accounts or network services.
"""

import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = PROJECT_ROOT / "data" / "sample.txt"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "observability_otel"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 2

QUESTION = "What is Retrieval-Augmented Generation and why is it useful?"

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


# --- Step 1: Configure OpenTelemetry Tracer ---
print_step(1, "Configure OpenTelemetry Tracer and Console Exporter")

# Create a TracerProvider and register the ConsoleSpanExporter
provider = TracerProvider()
processor = SimpleSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("ai_engineering.rag_service", "1.0.0")
print("OpenTelemetry Tracer configured: 'ai_engineering.rag_service'")


# --- Step 2: Index Sample Documents ---
print_step(2, "Index Chroma vector store")
loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=str(PERSIST_DIRECTORY))
print(f"Chroma store ready with {vector_store._collection.count()} chunks.")


# --- Step 3: Run Traced Pipeline ---
print_step(3, "Execute RAG pipeline with OpenTelemetry Span hierarchy")

# Root Span: covers the entire end-to-end request
with tracer.start_as_current_span("rag.request_pipeline") as root_span:
    root_span.set_attribute("gen_ai.operation.name", "rag_qa")
    root_span.set_attribute("user.query", QUESTION)
    print(f"Root Span started: 'rag.request_pipeline' (Trace ID: {root_span.get_span_context().trace_id:x})")

    # Child Span 1: Vector Database Retrieval
    with tracer.start_as_current_span("chroma.similarity_search") as retrieval_span:
        retrieval_start = time.perf_counter()
        retrieval_span.set_attribute("db.system", "chroma")
        retrieval_span.set_attribute("db.operation", "similarity_search")
        retrieval_span.set_attribute("retrieval.k", TOP_K)

        docs = vector_store.similarity_search(QUESTION, k=TOP_K)

        retrieval_duration = (time.perf_counter() - retrieval_start) * 1000
        retrieval_span.set_attribute("retrieval.results_count", len(docs))
        retrieval_span.set_attribute("retrieval.duration_ms", retrieval_duration)
        retrieval_span.add_event(
            "retrieval_completed",
            {"matched_doc_length": sum(len(d.page_content) for d in docs)},
        )
        print(f"  Child Span 1 finished: 'chroma.similarity_search' ({retrieval_duration:.2f} ms)")

    # Child Span 2: GenAI LLM Generation
    with tracer.start_as_current_span("gen_ai.chat_completion") as llm_span:
        gen_start = time.perf_counter()

        # Follow OpenTelemetry GenAI Semantic Conventions:
        llm_span.set_attribute("gen_ai.system", "huggingface")
        llm_span.set_attribute("gen_ai.request.model", "flan-t5-base")
        llm_span.set_attribute("gen_ai.prompt", QUESTION)

        # Context formatting
        context = "\n\n".join(d.page_content for d in docs)
        synthetic_answer = (
            "Retrieval-Augmented Generation improves LLM answers by first retrieving relevant chunks "
            "from an external knowledge base and then passing them as context to the generation model."
        )

        gen_duration = (time.perf_counter() - gen_start) * 1000
        llm_span.set_attribute("gen_ai.response.completion", synthetic_answer)
        llm_span.set_attribute("gen_ai.response.finish_reasons", ["stop"])
        llm_span.set_attribute("gen_ai.usage.prompt_tokens", len(context.split()))
        llm_span.set_attribute("gen_ai.usage.completion_tokens", len(synthetic_answer.split()))
        print(f"  Child Span 2 finished: 'gen_ai.chat_completion' ({gen_duration:.2f} ms)")

    root_span.set_attribute("rag.final_status", "success")

print_step(4, "OTEL Spans exported above in JSON format")
print("Notice how OpenTelemetry captures:")
print("  - Trace IDs linking all child operations to the same request root")
print("  - Standard GenAI semantic attributes ('gen_ai.system', 'gen_ai.request.model', tokens)")
print("  - Microsecond start and end timestamps for distributed APM dashboards.")
