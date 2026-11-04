"""Structured extraction: extracting from a retrieved document (RAG + extraction).

The earlier files in this folder extract from a short, hand-written string.
In practice, the text you want to extract from often lives inside a larger
document — a PDF, a report, a knowledge base article — and only a relevant
passage should be fed to the LLM. This file combines the retrieval step from
the RAG pipeline (chunks → embeddings → Chroma similarity search) with
the structured extraction step (PydanticOutputParser), showing how both
pieces fit together in a single chain.

The pipeline:
  1. Load and chunk data/sample.txt
  2. Embed chunks and store in Chroma
  3. Retrieve the most relevant chunk for a query
  4. Extract a typed MlConcept object from that chunk

No API key needed — uses the same local HuggingFace stack as every other
file in this project.
"""

import os
import shutil
from pathlib import Path

# Must be set before chromadb is imported.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class MlConcept(BaseModel):
    """A machine-learning concept extracted from a retrieved passage."""
    name: str = Field(description="Name of the ML concept or technique")
    one_line_definition: str = Field(
        description="A single sentence defining the concept in plain English"
    )
    related_terms: list[str] = Field(
        description="Up to three closely related terms or synonyms, as a list of strings"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = PROJECT_ROOT / "data" / "sample.txt"
PERSIST_DIRECTORY = PROJECT_ROOT / "db" / "extraction_rag"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "google/flan-t5-base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 1          # only one chunk — we want the single most relevant passage

QUERY = "What is overfitting and how does it relate to generalisation?"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

print_step(1, "Load and chunk the document")
loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
print(f"Loaded '{DOCUMENT_PATH.name}', split into {len(chunks)} chunks")

print_step(2, "Embed chunks and store in Chroma")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
vector_store = Chroma.from_documents(
    chunks, embeddings, persist_directory=str(PERSIST_DIRECTORY)
)
print(f"Stored {vector_store._collection.count()} chunks in '{PERSIST_DIRECTORY}'")

print_step(3, "Retrieve the most relevant chunk for the query")
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
retrieved_docs = retriever.invoke(QUERY)
retrieved_passage = retrieved_docs[0].page_content
print(f"Query: {QUERY!r}")
print(f"\nRetrieved passage:\n  {retrieved_passage}")

print_step(4, "Build the extraction chain")
parser = PydanticOutputParser(pydantic_object=MlConcept)
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 200},
)
prompt = PromptTemplate(
    template=(
        "Extract the main ML concept discussed in the passage below.\n"
        "{format_instructions}\n\n"
        "Passage: {passage}\n\n"
        "JSON:"
    ),
    input_variables=["passage"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
raw_chain = prompt | llm | StrOutputParser()

print_step(5, "Run extraction on the retrieved passage")
raw_output = raw_chain.invoke({"passage": retrieved_passage})
print(f"Raw model output:\n  {raw_output!r}")

print_step(6, "Parse into a typed MlConcept")
try:
    concept = parser.parse(raw_output)
    print("Parsed successfully!")
    print(f"  name:               {concept.name!r}")
    print(f"  one_line_definition: {concept.one_line_definition!r}")
    print(f"  related_terms:      {concept.related_terms}")
except Exception as e:
    print(
        f"Parsing failed ({type(e).__name__}): flan-t5-base may not follow the JSON schema "
        "over a long passage. A larger model handles this reliably."
    )
    print(f"\nRaw output was: {raw_output!r}")

print_step(7, "What this pipeline demonstrates")
print(
    "RAG and structured extraction compose cleanly:\n"
    "  retriever → relevant passage → LLM → PydanticOutputParser → typed object\n\n"
    "This is the foundation of document-processing pipelines that need to turn\n"
    "unstructured knowledge bases into structured, queryable records — without\n"
    "manually reading every document."
)
