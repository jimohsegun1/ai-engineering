"""Chain composition: the "stuff" documents chain.

create_stuff_documents_chain is the official building block for a pattern
you've already seen built by hand in rag-app/rag_pipeline.py: concatenate
every retrieved document into one prompt's {context} and ask the question
once. "Stuffing" only works while everything fits in the model's context
window — see 07_map_reduce_chain.py and 08_refine_chain.py for what to do
when it doesn't.
"""

from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


CHAT_MODEL = "google/flan-t5-base"
DOCS = [
    Document(page_content="The Eiffel Tower is located in Paris and is about 330 meters tall."),
    Document(page_content="Paris is the capital of France and its most populous city."),
]
QUESTION = "How tall is the Eiffel Tower?"

print_step(1, "Build the stuff-documents chain")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
)
prompt = PromptTemplate.from_template("Answer using only this context:\n\n{context}\n\nQuestion: {input}")
chain = create_stuff_documents_chain(llm, prompt)

print_step(2, "Invoke with all documents stuffed into one prompt")
print("Documents:\n" + "\n".join(f"  - {doc.page_content}" for doc in DOCS))
answer = chain.invoke({"context": DOCS, "input": QUESTION})
print(f"\nQuestion: {QUESTION}")
print(f"Answer: {answer}")
