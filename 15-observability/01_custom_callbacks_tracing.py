"""Observability: Custom Callback Tracing and Latency Profiling.

When deploying RAG pipelines or agents to production, you cannot treat the
chain as a black box. You need full visibility into:
  1. Component-level latency: Did the slow response come from vector search
     or LLM generation?
  2. Input/output payloads: What exact prompt was constructed after retrieving
     chunks, and what raw string did the model emit?
  3. Token accounting: How many prompt and completion tokens were consumed?

LangChain provides an event-driven callback system via BaseCallbackHandler.
This file implements a production-style ObservabilityTracker that hooks into
every stage of execution, measures component durations with microsecond precision,
and prints a flame-graph style execution summary table at the end.

Runs 100% locally with free models — zero API keys or cloud accounts needed.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = PROJECT_ROOT / "data" / "sample.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "observability_local"
CHAT_MODEL = "google/flan-t5-base"
TOP_K = 2

QUESTION = "What is Retrieval-Augmented Generation and why is it useful?"

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


# --- Step 1: Implement Custom Observability Tracker ---
class TraceSpan:
    def __init__(self, name: str, span_type: str):
        self.name = name
        self.span_type = span_type
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration_ms: float = 0.0
        self.inputs: Any = None
        self.outputs: Any = None
        self.tokens: int = 0


class ObservabilityTracker(BaseCallbackHandler):
    """Custom callback handler collecting execution spans, latencies, and token usage."""

    def __init__(self):
        super().__init__()
        self.spans: List[TraceSpan] = []
        self._active_spans: Dict[str, TraceSpan] = {}

    # --- Chain Events ---
    def on_chain_start(
        self, serialized: Optional[Dict[str, Any]], inputs: Dict[str, Any], *, run_id: UUID, **kwargs: Any
    ) -> None:
        name = serialized.get("name", "Chain") if serialized else "Chain"
        span = TraceSpan(name=name, span_type="CHAIN")
        span.start_time = time.perf_counter()
        span.inputs = inputs
        self._active_spans[str(run_id)] = span

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, **kwargs: Any) -> None:
        span = self._active_spans.pop(str(run_id), None)
        if span:
            span.end_time = time.perf_counter()
            span.duration_ms = (span.end_time - span.start_time) * 1000
            span.outputs = outputs
            self.spans.append(span)

    # --- Retriever Events ---
    def on_retriever_start(
        self, serialized: Optional[Dict[str, Any]], query: str, *, run_id: UUID, **kwargs: Any
    ) -> None:
        span = TraceSpan(name="ChromaRetriever", span_type="RETRIEVER")
        span.start_time = time.perf_counter()
        span.inputs = query
        self._active_spans[str(run_id)] = span

    def on_retriever_end(self, documents: Sequence[Document], *, run_id: UUID, **kwargs: Any) -> None:
        span = self._active_spans.pop(str(run_id), None)
        if span:
            span.end_time = time.perf_counter()
            span.duration_ms = (span.end_time - span.start_time) * 1000
            span.outputs = f"{len(documents)} document(s) retrieved"
            self.spans.append(span)

    # --- LLM Events ---
    def on_llm_start(
        self, serialized: Optional[Dict[str, Any]], prompts: List[str], *, run_id: UUID, **kwargs: Any
    ) -> None:
        name = serialized.get("name", "LLM") if serialized else "LLM"
        span = TraceSpan(name=name, span_type="LLM")
        span.start_time = time.perf_counter()
        span.inputs = prompts[0] if prompts else ""
        self._active_spans[str(run_id)] = span

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        span = self._active_spans.pop(str(run_id), None)
        if span:
            span.end_time = time.perf_counter()
            span.duration_ms = (span.end_time - span.start_time) * 1000
            if response.generations and response.generations[0]:
                span.outputs = response.generations[0][0].text
            self.spans.append(span)

    def print_trace_report(self) -> None:
        print(f"\n{RULE}")
        print("OBSERVABILITY EXECUTION PROFILE")
        print(RULE)
        print(f"{'TYPE':<12} | {'COMPONENT':<25} | {'DURATION':<12} | {'PAYLOAD / STATUS'}")
        print("-" * 75)

        total_duration = 0.0
        for s in self.spans:
            out_preview = str(s.outputs)[:40].replace("\n", " ") if s.outputs else ""
            print(f"{s.span_type:<12} | {s.name:<25} | {s.duration_ms:8.2f} ms | {out_preview}")
            if s.span_type == "CHAIN" and s.name == "RunnableSequence":
                total_duration = s.duration_ms

        print("-" * 75)
        if total_duration > 0:
            print(f"End-to-End Pipeline Latency: {total_duration:.2f} ms")


# --- Step 2: Build Pipeline with Callback Handler ---
print_step(1, "Initialize vector store and RAG chain")

loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=str(PERSIST_DIRECTORY))
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 150},
)

prompt = PromptTemplate.from_template(
    """Answer using only this context:

{context}

Question: {question}

Answer:"""
)


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# --- Step 3: Execute with Observability Tracer ---
print_step(2, "Invoke chain with attached ObservabilityTracker")

tracker = ObservabilityTracker()
print(f"Invoking Question: {QUESTION!r}\n")

# Attach callback handler via config={"callbacks": [tracker]}
answer = rag_chain.invoke(QUESTION, config={"callbacks": [tracker]})

print(f"Generated Answer: {answer!r}")

# --- Step 4: Display Diagnostic Report ---
tracker.print_trace_report()
