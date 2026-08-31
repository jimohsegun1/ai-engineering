"""RAG (Retrieval-Augmented Generation) app built with LangChain.

Steps:
1. Prepare input document
2. Chunking
3. Create embeddings
4. Store embeddings in vector database
5. Similarity search
6. RAG pipeline
"""

import os
import shutil

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DOCUMENT_PATH = "data/sample.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "chroma_db"
QOREBIT_BASE_URL = "https://api.qorebit.ai/v1"
CHAT_MODEL = "openai/gpt-4o"
TOP_K = 3

QUESTION = "What is Retrieval-Augmented Generation and why is it useful?"


# --- Step 1: Prepare input document ---
# Load the raw text file into LangChain's Document format (text + metadata).
loader = TextLoader(DOCUMENT_PATH, encoding="utf-8")
documents = loader.load()
print(f"[1] Loaded {len(documents)} document(s) from {DOCUMENT_PATH}")


# --- Step 2: Chunking ---
# Split the document into overlapping chunks so retrieval can return
# focused passages instead of the whole file.
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
print(f"[2] Split into {len(chunks)} chunks")


# --- Step 3: Create embeddings ---
# An embedding model turns text into a vector of numbers that captures
# meaning, so similar text ends up with similar vectors. This uses a
# free local model since Qorebit's docs only cover chat completions.
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
print(f"[3] Loaded embedding model: {EMBEDDING_MODEL}")


# --- Step 4: Store embeddings in vector database ---
# Chroma embeds every chunk and stores the vectors (plus original text)
# on disk so they can be searched later without re-embedding. We wipe
# any previous run's data first so re-running this script doesn't keep
# appending duplicate chunks to the same collection.
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY,
)
print(f"[4] Stored {vector_store._collection.count()} chunks in '{PERSIST_DIRECTORY}'")


# --- Step 5: Similarity search ---
# Embed the query and ask the vector database for the closest chunks.
results = vector_store.similarity_search(QUESTION, k=TOP_K)
print(f"[5] Top {TOP_K} chunks for: {QUESTION!r}")
for i, doc in enumerate(results):
    print(f"    result {i}: {doc.page_content[:100]}...")


# --- Step 6: RAG pipeline ---
# Wire retrieval + prompt + LLM together: retrieve relevant chunks,
# inject them as context, and have the LLM generate a grounded answer.
def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0,
    api_key=os.environ["QOREBIT_API_KEY"],
    base_url=QOREBIT_BASE_URL,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "AI Engineering RAG Learning App",
        # Qorebit's WAF blocks the openai SDK's default User-Agent string,
        # so we identify our app instead.
        "User-Agent": "rag-app-learning-project/1.0",
    },
)
prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.
If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke(QUESTION)
print(f"\n[6] Question: {QUESTION}")
print(f"[6] Answer: {answer}")
