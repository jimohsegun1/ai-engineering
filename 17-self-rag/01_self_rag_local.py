"""Self-RAG (Corrective RAG) using LangGraph: local, free model edition.

In standard RAG, the retriever blindly passes whatever chunks it found
directly to the LLM. If the query was phrased poorly or retrieval fetched
irrelevant context, the LLM hallucinates or generates a poor answer.

Self-RAG (and Corrective RAG / CRAG) introduces an active evaluation loop:
  1. Retrieve documents from the vector store.
  2. Grade retrieved documents for relevance to the query.
  3. Conditional Routing:
     - If relevant documents exist -> proceed to Generation.
     - If documents are irrelevant -> Rewrite the query and re-retrieve.
     - Loop protection: caps rewrites to MAX_ITERATIONS to prevent infinite loops.
  4. Generate the final answer from filtered relevant context.

Runs locally on sentence-transformers embeddings + google/flan-t5-base.
No API key or external service needed.
"""

import os
import shutil
import textwrap
from pathlib import Path
from typing import List, TypedDict

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = PROJECT_ROOT / "data" / "sample.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "self_rag_local"
CHAT_MODEL = "google/flan-t5-base"
TOP_K = 3
MAX_RETRIES = 1

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


def print_passage(label: str, text: str) -> None:
    print(f"\n  {label}")
    print("  " + "-" * 60)
    wrapped = textwrap.fill(text, width=68, initial_indent="  ", subsequent_indent="  ")
    print(wrapped)


# --- Step 1: Set up Vector Store ---
print_step(1, "Prepare vector store with sample documents")
loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=str(PERSIST_DIRECTORY))
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
print(f"Indexed {vector_store._collection.count()} chunks in '{PERSIST_DIRECTORY}'")


# --- Step 2: Initialize LLM and Prompts ---
print_step(2, "Initialize local LLM and grading/rewriting prompts")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 150},
)

# Document grader prompt: evaluates if a document chunk is relevant to the question
grader_prompt = PromptTemplate.from_template(
    """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keywords or semantic meaning related to the user question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.

Document:
{document}

Question: {question}

Is the document relevant to the question? (yes or no):"""
)
grader_chain = grader_prompt | llm | StrOutputParser()

# Query rewriter prompt: improves query formulation when retrieval fails
rewriter_prompt = PromptTemplate.from_template(
    """You are an AI that rewrites user questions to optimize retrieval from a technical vector store.
Look at the input question and formulate an improved keyword-rich query that retrieves better technical passages.

Initial question: {question}

Improved question:"""
)
rewriter_chain = rewriter_prompt | llm | StrOutputParser()

# Answer generator prompt
generator_prompt = PromptTemplate.from_template(
    """Answer the question using only the supporting context below.
If you do not know the answer, say you do not know.

Context:
{context}

Question: {question}

Answer:"""
)
generator_chain = generator_prompt | llm | StrOutputParser()


# --- Step 3: Define Graph State and Nodes ---
class GraphState(TypedDict):
    question: str
    documents: List[str]
    filtered_documents: List[str]
    generation: str
    retries: int


def retrieve_node(state: GraphState) -> dict:
    """Retrieve documents matching current question."""
    q = state["question"]
    print(f"\n[Node: Retrieve] Fetching top {TOP_K} chunks for query: {q!r}")
    docs = retriever.invoke(q)
    return {"documents": [d.page_content for d in docs]}


def grade_documents_node(state: GraphState) -> dict:
    """Filter retrieved chunks, keeping only relevant ones."""
    q = state["question"]
    docs = state["documents"]
    filtered = []
    print(f"\n[Node: Grade Documents] Grading {len(docs)} chunks for relevance...")

    for i, doc in enumerate(docs, start=1):
        grade = grader_chain.invoke({"question": q, "document": doc}).strip().lower()
        is_relevant = "yes" in grade
        status = "KEEP (relevant)" if is_relevant else "DROP (irrelevant)"
        print(f"  Chunk {i}: {status} -> Model grade: {grade!r}")
        if is_relevant:
            filtered.append(doc)

    return {"filtered_documents": filtered}


def transform_query_node(state: GraphState) -> dict:
    """Rewrite query to optimize subsequent retrieval."""
    old_q = state["question"]
    current_retries = state.get("retries", 0)
    print(f"\n[Node: Rewrite Query] No relevant docs found! Rewriting query (Attempt {current_retries + 1}/{MAX_RETRIES})...")
    new_q = rewriter_chain.invoke({"question": old_q}).strip()
    print(f"  Old query: {old_q!r}")
    print(f"  Rewritten query: {new_q!r}")
    return {"question": new_q, "retries": current_retries + 1}


def generate_node(state: GraphState) -> dict:
    """Generate final response based on filtered context."""
    q = state["question"]
    docs = state.get("filtered_documents") or state.get("documents") or []
    print(f"\n[Node: Generate] Generating answer from {len(docs)} context chunk(s)...")
    context = "\n\n".join(docs)
    answer = generator_chain.invoke({"context": context, "question": q}).strip()
    return {"generation": answer}


def decide_to_generate(state: GraphState) -> str:
    """Conditional edge: route to generate or rewrite query."""
    filtered = state.get("filtered_documents", [])
    retries = state.get("retries", 0)

    if filtered:
        print("[Decision] At least one chunk is relevant -> Proceed to Generate.")
        return "generate"

    if retries >= MAX_RETRIES:
        print(f"[Decision] Max retries ({MAX_RETRIES}) reached -> Fallback to Generate.")
        return "generate"

    print("[Decision] All chunks irrelevant -> Proceed to Transform Query.")
    return "transform_query"


# --- Step 4: Build and Compile StateGraph ---
print_step(3, "Assemble LangGraph state machine")
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("transform_query", transform_query_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "transform_query": "transform_query",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END)

app = workflow.compile()


# --- Step 5: Test Execution ---
print_step(4, "Run Self-RAG loop with a sample question")
test_question = "What is Retrieval-Augmented Generation and why is it useful?"
print(f"Starting question: {test_question!r}\n")

initial_state: GraphState = {
    "question": test_question,
    "documents": [],
    "filtered_documents": [],
    "generation": "",
    "retries": 0,
}

final_state = app.invoke(initial_state)

print(f"\n{RULE}")
print("SELF-RAG EXECUTION FINISHED")
print(RULE)
print(f"Final Question: {final_state['question']}")
print(f"Total Rewrites: {final_state['retries']}")
print(f"Retained Docs:  {len(final_state['filtered_documents'])}")
print_passage("Final Answer", final_state["generation"])
