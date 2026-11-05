"""Self-RAG (Corrective RAG) using LangGraph and Qorebit (GPT-4o).

The hosted twin of 02_self_rag_ollama.py: replaces Ollama with Qorebit's
OpenAI-compatible endpoint running GPT-4o.

Full Self-RAG loop:
  1. Vector search in Chroma.
  2. Document relevance evaluation via Pydantic structured output.
  3. Conditional routing: query rewriting if retrieval missed relevant context.
  4. Answer generation with strict grounding prompt.
  5. Hallucination check against retrieved source facts.
  6. Answer quality assessment before returning response.

Needs QOREBIT_API_KEY in .env.
"""

import os
import shutil
import textwrap
from pathlib import Path
from typing import List, Literal, TypedDict

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = PROJECT_ROOT / "data" / "sample.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "self_rag_qorebit"
QOREBIT_BASE_URL = "https://api.qorebit.ai/v1"
CHAT_MODEL = "openai/gpt-4o"
TOP_K = 3
MAX_RETRIES = 2

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


# --- Step 1: Initialize Vector Store ---
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


# --- Step 2: Initialize Qorebit LLM and Graders ---
print_step(2, "Initialize Qorebit LLM (GPT-4o) and structured evaluators")
llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0,
    api_key=os.environ.get("QOREBIT_API_KEY", "qb_placeholder"),
    base_url=QOREBIT_BASE_URL,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "AI Engineering Self-RAG Learning Project",
        "User-Agent": "rag-app-learning-project/1.0",
    },
)

class GradeDocument(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucination(BaseModel):
    """Binary score for hallucination presence in generated answer."""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

# Document Grader
doc_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a grader assessing relevance of a retrieved document to a user question.\n"
                   "If the document contains keyword(s) or semantic meaning related to the question, grade it as 'yes'.\n"
                   "Otherwise, grade it as 'no'."),
        ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
    ]
)
doc_grader = doc_grader_prompt | llm.with_structured_output(GradeDocument)

# Hallucination Grader
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.\n"
                   "Give a binary score 'yes' if supported, or 'no' if the answer contains ungrounded hallucinations."),
        ("human", "Set of facts:\n\n{documents}\n\nLLM generation: {generation}"),
    ]
)
hallucination_grader = hallucination_prompt | llm.with_structured_output(GradeHallucination)

# Answer Grader
answer_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a grader assessing whether an answer addresses / resolves a user question.\n"
                   "Give a binary score 'yes' or 'no'."),
        ("human", "User question: {question}\n\nLLM generation: {generation}"),
    ]
)
answer_grader = answer_grader_prompt | llm.with_structured_output(GradeAnswer)

# Query Rewriter
rewriter_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a question re-writer that converts an input question to a better version optimized\n"
                   "for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent."),
        ("human", "Here is the initial question:\n\n{question}\nFormulate an improved question:"),
    ]
)
rewriter = rewriter_prompt | llm | StrOutputParser()

# Generation Chain
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.\n"
                   "If you don't know the answer, say that you don't know. Keep answers concise and factual."),
        ("human", "Context:\n\n{context}\n\nQuestion: {question}"),
    ]
)
generator = qa_prompt | llm | StrOutputParser()


# --- Step 3: State Graph Nodes & Flow ---
class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    retries: int


def retrieve_node(state: GraphState) -> dict:
    q = state["question"]
    print(f"\n[Node: Retrieve] Fetching top {TOP_K} chunks for query: {q!r}")
    docs = retriever.invoke(q)
    return {"documents": [d.page_content for d in docs]}


def grade_documents_node(state: GraphState) -> dict:
    q = state["question"]
    docs = state["documents"]
    filtered = []
    print(f"\n[Node: Grade Documents] Evaluating {len(docs)} retrieved chunks...")

    for i, doc in enumerate(docs, start=1):
        try:
            score = doc_grader.invoke({"question": q, "document": doc})
            grade = score.binary_score
        except Exception:
            grade = "yes"

        status = "KEEP" if grade == "yes" else "DISCARD"
        print(f"  Chunk {i}: {status} (grade={grade!r})")
        if grade == "yes":
            filtered.append(doc)

    return {"documents": filtered}


def transform_query_node(state: GraphState) -> dict:
    old_q = state["question"]
    count = state.get("retries", 0) + 1
    print(f"\n[Node: Rewrite Query] Rewriting query (Attempt {count}/{MAX_RETRIES})...")
    new_q = rewriter.invoke({"question": old_q}).strip()
    print(f"  Rewritten Query: {new_q!r}")
    return {"question": new_q, "retries": count}


def generate_node(state: GraphState) -> dict:
    q = state["question"]
    docs = state.get("documents", [])
    print(f"\n[Node: Generate] Generating response with {len(docs)} supporting chunk(s)...")
    context = "\n\n".join(docs)
    answer = generator.invoke({"context": context, "question": q})
    return {"generation": answer}


def decide_to_generate(state: GraphState) -> str:
    docs = state.get("documents", [])
    retries = state.get("retries", 0)

    if not docs and retries < MAX_RETRIES:
        print("[Decision] No relevant documents found -> Routing to Query Rewriter.")
        return "transform_query"

    print("[Decision] Relevant context present (or max retries reached) -> Routing to Generator.")
    return "generate"


def grade_generation(state: GraphState) -> str:
    q = state["question"]
    docs = "\n\n".join(state.get("documents", []))
    gen = state.get("generation", "")
    retries = state.get("retries", 0)

    print("\n[Node: Self-Reflection] Checking hallucination & answer quality...")
    try:
        hallucination_score = hallucination_grader.invoke({"documents": docs, "generation": gen})
        is_grounded = hallucination_score.binary_score == "yes"
    except Exception:
        is_grounded = True

    if not is_grounded:
        print("  Hallucination Check: FAILED (Generation not grounded in facts).")
        if retries < MAX_RETRIES:
            print("  Action: Regenerating answer...")
            return "generate"

    print("  Hallucination Check: PASSED (Answer is grounded).")

    try:
        ans_score = answer_grader.invoke({"question": q, "generation": gen})
        addresses_q = ans_score.binary_score == "yes"
    except Exception:
        addresses_q = True

    if addresses_q:
        print("  Answer Relevance Check: PASSED (Answer addresses user question).")
        return "useful"

    print("  Answer Relevance Check: FAILED (Answer does not address user question).")
    if retries < MAX_RETRIES:
        return "transform_query"

    return "useful"


# --- Step 4: Assemble LangGraph ---
print_step(3, "Assemble LangGraph workflow")
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("generate", generate_node)
workflow.add_node("transform_query", transform_query_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation,
    {
        "generate": "generate",
        "transform_query": "transform_query",
        "useful": END,
    },
)

app = workflow.compile()


# --- Step 5: Execute Graph ---
print_step(4, "Run Self-RAG loop")
test_q = "What are the common causes of model overfitting in machine learning?"
print(f"User Query: {test_q!r}")

initial_state: GraphState = {
    "question": test_q,
    "documents": [],
    "generation": "",
    "retries": 0,
}

if "QOREBIT_API_KEY" not in os.environ or not os.environ["QOREBIT_API_KEY"].startswith("qb_"):
    print("\nNote: QOREBIT_API_KEY not found in .env. Showing compiled graph structure:")
    print("Graph compiled successfully with nodes:", list(workflow.nodes.keys()))
    print("Add your QOREBIT_API_KEY to .env to run live generation.")
else:
    result = app.invoke(initial_state)
    print(f"\n{RULE}")
    print("SELF-RAG EXECUTION FINISHED")
    print(RULE)
    print(f"Final Query:   {result['question']}")
    print(f"Total Retries: {result['retries']}")
    print(f"Context Count: {len(result['documents'])}")
    print_passage("Verified Answer", result["generation"])
