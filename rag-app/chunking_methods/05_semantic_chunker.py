"""RAG chunking method: SemanticChunker (langchain-experimental).

Instead of splitting by a fixed size, this embeds each sentence and cuts
a new chunk wherever the semantic similarity between consecutive
sentences drops sharply — the idea being that a big meaning-shift is a
better place to split than an arbitrary character count.

This blurs the usual step 2/3 boundary: semantic chunking needs an
embedding model *during* chunking, not after it. So step 2 below creates
the embedding model early (normally step 3's job) in order to chunk,
and step 3 just points out that the same model is being reused.

Steps:
1. Prepare input document
2. Chunking            <- SemanticChunker (embeddings created here too)
3. Create embeddings   <- reuses the model already created in step 2
4. Store embeddings in vector database
5. Similarity search
6. RAG pipeline (commented out — uncomment when you're ready to call Qorebit)
"""

import os
import shutil
import textwrap

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

DOCUMENT_PATH = "data/sample.txt"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "db/chunking_semantic"
QOREBIT_BASE_URL = "https://api.qorebit.ai/v1"
CHAT_MODEL = "openai/gpt-4o"
TOP_K = 3

QUESTION = "What is Retrieval-Augmented Generation and why is it useful?"

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


def print_passage(label: str, text: str, char_count: int | None = None) -> None:
    suffix = f" ({char_count} chars)" if char_count is not None else ""
    print(f"\n  {label}{suffix}")
    print("  " + "-" * 60)
    wrapped = textwrap.fill(text, width=68, initial_indent="  ", subsequent_indent="  ")
    print(wrapped)


# --- Step 1: Prepare input document ---
print_step(1, "Prepare input document")
loader = TextLoader(DOCUMENT_PATH, encoding="utf-8")
documents = loader.load()
print(f"Loaded {len(documents)} document(s) from {DOCUMENT_PATH}")


# --- Step 2: Chunking (SemanticChunker) ---
# Needs an embedding model to measure meaning-shifts between sentences,
# so we create it here rather than in step 3 as the other files do.
#
# breakpoint_threshold_amount defaults to 95 (only the most extreme 5%
# of meaning-shifts count as a split point), which is tuned for long
# documents with many sentences. Our short sample doc only has ~10
# sentences, so the default collapses almost everything into one giant
# chunk. Lowering it to 50 makes the splitter sensitive enough to find
# multiple breakpoints in a short document like this one.
#
# SemanticChunker can also emit an empty trailing chunk depending on
# the document and threshold — we filter those out defensively.
print_step(2, "Chunking (SemanticChunker)")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=50,
)
chunks = [c for c in splitter.split_documents(documents) if c.page_content.strip()]
print(f"Split into {len(chunks)} chunks")
for i, chunk in enumerate(chunks, start=1):
    print_passage(f"Chunk {i}/{len(chunks)}", chunk.page_content, len(chunk.page_content))


# --- Step 3: Create embeddings ---
# Already created above (step 2 needed it) — reusing the same model here.
print_step(3, "Create embeddings")
print(f"Reusing embedding model already loaded in step 2: {EMBEDDING_MODEL}")


# --- Step 4: Store embeddings in vector database ---
print_step(4, "Store embeddings in vector database")
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY,
)
print(f"Stored {vector_store._collection.count()} chunks in '{PERSIST_DIRECTORY}'")


# --- Step 5: Similarity search ---
print_step(5, "Similarity search")
results = vector_store.similarity_search(QUESTION, k=TOP_K)
print(f"Query: {QUESTION!r}")
print(f"Top {TOP_K} matching chunks:")
for i, doc in enumerate(results, start=1):
    print_passage(f"Result {i}/{TOP_K}", doc.page_content)


# --- Step 6: RAG pipeline ---
# Commented out on purpose — uncomment when you're ready to call Qorebit.
# print_step(6, "RAG pipeline")
#
#
# def format_docs(docs) -> str:
#     return "\n\n".join(doc.page_content for doc in docs)
#
#
# retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
# llm = ChatOpenAI(
#     model=CHAT_MODEL,
#     temperature=0,
#     api_key=os.environ["QOREBIT_API_KEY"],
#     base_url=QOREBIT_BASE_URL,
#     default_headers={
#         "HTTP-Referer": "http://localhost",
#         "X-Title": "AI Engineering RAG Learning App",
#         "User-Agent": "rag-app-learning-project/1.0",
#     },
# )
# prompt = ChatPromptTemplate.from_template(
#     """Answer the question using only the context below.
# If the context doesn't contain the answer, say you don't know.
#
# Context:
# {context}
#
# Question: {question}
#
# Answer:"""
# )
#
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )
#
# answer = rag_chain.invoke(QUESTION)
# print(f"Question: {QUESTION}")
# print_passage("Answer", answer)
