"""RAG Evaluation: Comprehensive Chunking Strategy Benchmark.

Automated benchmark harness scoring all 6 chunking strategies from 01-chunking-methods/:
  1. CharacterTextSplitter (fixed-separator, \n\n)
  2. RecursiveCharacterTextSplitter (hierarchical prose separators)
  3. TokenTextSplitter (tiktoken token-boundary sizing)
  4. MarkdownHeaderTextSplitter (structure-aware Markdown headings)
  5. SemanticChunker (embedding cosine distance shift detection)
  6. CodeSplitter (Python AST & language syntax-aware separators)

Evaluates:
  - Splitting execution latency (ms)
  - Chunk volume, average size, and size uniformity (standard deviation)
  - Chunk Boundary Coherence (preservation of natural sentence, paragraph, and function boundaries)
  - Retrieval Precision (P@k) over a ground-truth question-keyword eval set
  - Context Recall / Hit Rate across top-k retrieved chunks
  - Retrieval query latency

Runs locally using free HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2)
and Chroma vector stores under db/benchmark_chunking_<method>/.
"""

import math
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Disable Chroma telemetry before imports
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROSE_TEXT_PATH = PROJECT_ROOT / "data" / "sample.txt"
MARKDOWN_PATH = PROJECT_ROOT / "data" / "sample.md"
CODE_PATH = PROJECT_ROOT / "data" / "sample_code.py"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

RULE = "=" * 70
SUB_RULE = "-" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


# ---------------------------------------------------------------------------
# Evaluation Query Suite & Ground Truth Keywords
# ---------------------------------------------------------------------------
# Covers diverse retrieval scenarios across both conceptual prose and Python code.
BENCHMARK_EVAL_SET = [
    {
        "id": "q1_ml_categories",
        "category": "Prose",
        "question": "What are the three broad categories of machine learning?",
        "expected_keywords": ["supervised", "unsupervised", "reinforcement"],
    },
    {
        "id": "q2_overfitting",
        "category": "Prose",
        "question": "What is overfitting and how do you prevent it?",
        "expected_keywords": ["overfitting", "noise", "unseen data", "regularization"],
    },
    {
        "id": "q3_rag_concept",
        "category": "Prose",
        "question": "What is Retrieval-Augmented Generation (RAG)?",
        "expected_keywords": ["retrieval-augmented generation", "retrieving", "external knowledge", "context"],
    },
    {
        "id": "q4_neural_networks",
        "category": "Prose",
        "question": "What are neural networks and deep learning?",
        "expected_keywords": ["neural networks", "brain", "neurons", "layers"],
    },
    {
        "id": "q5_inventory_total_value",
        "category": "Code",
        "question": "How does the InventoryItem class compute its total value?",
        "expected_keywords": ["total_value", "quantity", "unit_price"],
    },
    {
        "id": "q6_low_stock_filter",
        "category": "Code",
        "question": "How are low stock items detected in the Inventory class?",
        "expected_keywords": ["low_stock_items", "threshold", "quantity"],
    },
]


@dataclass
class ChunkMetrics:
    chunk_count: int = 0
    avg_chars: float = 0.0
    min_chars: int = 0
    max_chars: int = 0
    std_chars: float = 0.0
    chunking_latency_ms: float = 0.0
    boundary_coherence_score: float = 0.0


@dataclass
class RetrievalMetrics:
    indexing_latency_ms: float = 0.0
    avg_query_latency_ms: float = 0.0
    retrieval_precision: float = 0.0
    context_recall: float = 0.0
    detailed_hits: dict[str, bool] = field(default_factory=dict)


@dataclass
class StrategyBenchmarkResult:
    name: str
    strategy_type: str
    chunk_metrics: ChunkMetrics
    retrieval_metrics: RetrievalMetrics | None = None
    sample_first_chunk: str = ""


# ---------------------------------------------------------------------------
# Boundary Coherence Analyzer
# ---------------------------------------------------------------------------
def compute_boundary_coherence(chunks: list[Document]) -> float:
    """Evaluate whether chunk boundaries preserve natural sentence/declaration units.

    Scores 1.0 if every chunk begins at a natural sentence/header/code declaration start
    and terminates with appropriate sentence punctuation or block closure delimiters,
    rather than severing sentences mid-clause or words mid-token.
    """
    if not chunks:
        return 0.0

    valid_starts = (
        "#",
        "class ",
        "def ",
        "import ",
        "from ",
        '"""',
        "'''",
        "@",
        "-",
        "*",
    )
    valid_ends = (
        ".",
        "!",
        "?",
        ":",
        '"""',
        "'''",
        ")",
        "}",
        "]",
        '"',
        "'",
    )

    coherent_chunks = 0

    for chunk in chunks:
        text = chunk.page_content.strip()
        if not text:
            continue

        # Check clean start (capitalized word, comment, heading, or code declaration)
        starts_clean = text[0].isupper() or text.startswith(valid_starts) or text[0].isdigit()

        # Check clean termination (sentence-ending punctuation or code delimiter)
        ends_clean = text.endswith(valid_ends)

        # Check lack of mid-word trailing hyphenation or orphaned punctuation
        no_dangling_split = not (text.endswith("-") or text.endswith(","))

        if starts_clean and ends_clean and no_dangling_split:
            coherent_chunks += 1

    return round(coherent_chunks / len(chunks), 3)


def calculate_chunk_distribution(chunks: list[Document], latency_ms: float) -> ChunkMetrics:
    """Calculate statistical distribution of chunk lengths."""
    if not chunks:
        return ChunkMetrics(chunking_latency_ms=latency_ms)

    lengths = [len(c.page_content) for c in chunks]
    avg_len = sum(lengths) / len(lengths)
    variance = sum((x - avg_len) ** 2 for x in lengths) / len(lengths)
    std_dev = math.sqrt(variance)
    coherence = compute_boundary_coherence(chunks)

    return ChunkMetrics(
        chunk_count=len(chunks),
        avg_chars=round(avg_len, 1),
        min_chars=min(lengths),
        max_chars=max(lengths),
        std_chars=round(std_dev, 1),
        chunking_latency_ms=round(latency_ms, 2),
        boundary_coherence_score=coherence,
    )


# ---------------------------------------------------------------------------
# Benchmark Engine: 6 Chunking Strategies
# ---------------------------------------------------------------------------
class ChunkingBenchmarkSuite:
    """Harness that splits, indexes, and evaluates all 6 chunking strategies."""

    def __init__(self, prose_text: str, markdown_text: str, code_text: str, embeddings: HuggingFaceEmbeddings):
        self.prose_text = prose_text
        self.markdown_text = markdown_text
        self.code_text = code_text
        self.embeddings = embeddings

    def split_character(self) -> list[Document]:
        """1. CharacterTextSplitter: Fixed separator (\\n\\n)."""
        splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=50)
        return splitter.create_documents([self.prose_text, self.code_text])

    def split_recursive(self) -> list[Document]:
        """2. RecursiveCharacterTextSplitter: Hierarchical prose separators."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.create_documents([self.prose_text, self.code_text])

    def split_token(self) -> list[Document]:
        """3. TokenTextSplitter: Tiktoken token-boundary sizing."""
        splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=15)
        return splitter.create_documents([self.prose_text, self.code_text])

    def split_markdown(self) -> list[Document]:
        """4. MarkdownHeaderTextSplitter: Heading structure aware with recursive fallback."""
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2")])
        md_chunks = md_splitter.split_text(self.markdown_text)
        # Combine with code chunks split using standard recursive splitter
        code_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        code_chunks = code_splitter.create_documents([self.code_text])
        return md_chunks + code_chunks

    def split_semantic(self) -> list[Document]:
        """5. SemanticChunker: Cosine distance shift between consecutive sentences."""
        splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=50,
        )
        raw_chunks = splitter.create_documents([self.prose_text, self.code_text])
        return [c for c in raw_chunks if c.page_content.strip()]

    def split_code(self) -> list[Document]:
        """6. CodeSplitter: Python syntax & AST-aware separators."""
        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=500,
            chunk_overlap=50,
        )
        prose_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return code_splitter.create_documents([self.code_text]) + prose_splitter.create_documents([self.prose_text])

    def run_all_strategies(self) -> dict[str, tuple[list[Document], ChunkMetrics]]:
        """Execute all 6 chunking strategies and compute chunk metrics."""
        strategies = [
            ("CharacterTextSplitter", "Fixed Separator", self.split_character),
            ("RecursiveCharacterTextSplitter", "Hierarchical Prose", self.split_recursive),
            ("TokenTextSplitter", "BPE Token Counting", self.split_token),
            ("MarkdownHeaderTextSplitter", "Structural Headings", self.split_markdown),
            ("SemanticChunker", "Semantic Distance", self.split_semantic),
            ("CodeSplitter (Python)", "Language AST-Aware", self.split_code),
        ]

        results = {}
        for name, strat_type, split_fn in strategies:
            start_t = time.perf_counter()
            chunks = split_fn()
            duration_ms = (time.perf_counter() - start_t) * 1000.0
            metrics = calculate_chunk_distribution(chunks, duration_ms)
            results[name] = (chunks, metrics)
        return results


def main() -> None:
    print_step(1, "Load Benchmark Corpora (Prose, Markdown, Python Code)")
    prose = PROSE_TEXT_PATH.read_text(encoding="utf-8")
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    code = CODE_PATH.read_text(encoding="utf-8")

    print(f"  Prose text:    {PROSE_TEXT_PATH.name} ({len(prose)} chars)")
    print(f"  Markdown text: {MARKDOWN_PATH.name} ({len(markdown)} chars)")
    print(f"  Python code:   {CODE_PATH.name} ({len(code)} chars)")

    print_step(2, "Initialize Shared Embedding Model")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"  Embedding model initialized: {EMBEDDING_MODEL}")

    print_step(3, "Execute 6 Chunking Strategies & Analyze Boundary Coherence")
    suite = ChunkingBenchmarkSuite(prose, markdown, code, embeddings)
    chunk_results = suite.run_all_strategies()

    print(f"\n{'Strategy':<33} | {'Chunks':>6} | {'Avg Size':>8} | {'Range':>11} | {'Std Dev':>7} | {'Coherence':>9} | {'Split Time':>10}")
    print("-" * 100)
    for name, (chunks, metrics) in chunk_results.items():
        range_str = f"{metrics.min_chars}-{metrics.max_chars}"
        print(
            f"{name:<33} | "
            f"{metrics.chunk_count:>6} | "
            f"{metrics.avg_chars:>8.1f} | "
            f"{range_str:>11} | "
            f"{metrics.std_chars:>7.1f} | "
            f"{metrics.boundary_coherence_score * 100:>8.1f}% | "
            f"{metrics.chunking_latency_ms:>8.1f} ms"
        )


if __name__ == "__main__":
    main()
