"""Chain composition: a conversational (history-aware) retrieval chain.

09_retrieval_chain.py answers one question at a time with no memory of what
came before, so a follow-up like "why is it useful?" would be searched for
literally instead of about whatever "it" refers to. create_history_aware_retriever
fixes that: it first rewrites the new question into a standalone one using
the chat history, *then* retrieves — the modern replacement for the legacy
ConversationalRetrievalChain. Runs entirely on the free local flan-t5-base +
Chroma stack used throughout this project.
"""

import os
import shutil
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = PROJECT_ROOT / "data" / "sample.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "chains_conversational"
CHAT_MODEL = "google/flan-t5-base"
TOP_K = 3

TURNS = [
    "What is Retrieval-Augmented Generation?",
    "Why is it useful?",
]

print_step(1, "Load, chunk, and embed the document")
loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=str(PERSIST_DIRECTORY))
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
print(f"Stored {vector_store._collection.count()} chunks in '{PERSIST_DIRECTORY}'")

print_step(2, "Build the history-aware retriever (rewrites follow-ups before retrieving)")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
)
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Given the chat history and a follow-up question, rewrite the follow-up "
                   "question to be a standalone question. Return only the rewritten question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

print_step(3, "Build the full conversational retrieval chain")
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question using only this context:\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
conversational_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

print_step(4, "Run the conversation turn by turn")
chat_history = []
for i, question in enumerate(TURNS, start=1):
    print(f"\nTurn {i}")
    print(f"  User: {question}")
    result = conversational_chain.invoke({"input": question, "chat_history": chat_history})
    answer = result["answer"]
    print(f"  AI:   {answer}")
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))
