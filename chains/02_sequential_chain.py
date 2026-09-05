"""LCEL chain composition: sequential chaining.

Feeds one chain's output into a second chain's input, using
RunnablePassthrough.assign to keep every intermediate value (the original
text, the summary, and the final title) available in the result instead of
throwing earlier steps away.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


CHAT_MODEL = "google/flan-t5-base"
TEXT = (
    "Vector databases store embeddings and use indexing structures like HNSW to find "
    "the closest matches to a query vector almost instantly, even across millions of "
    "stored vectors, which makes them well suited for retrieval-augmented generation."
)

print_step(1, "Build the two chains")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
)
summarize_prompt = PromptTemplate.from_template("Summarize this text in one sentence:\n\n{text}")
title_prompt = PromptTemplate.from_template("Write a short title (max 6 words) for this summary:\n\n{summary}")

summarize_chain = summarize_prompt | llm | StrOutputParser()
title_chain = RunnableLambda(lambda x: {"summary": x["summary"]}) | title_prompt | llm | StrOutputParser()

print_step(2, "Chain them: text -> summary -> title")
full_chain = (
    RunnablePassthrough.assign(summary=summarize_chain)
    | RunnablePassthrough.assign(title=title_chain)
)

result = full_chain.invoke({"text": TEXT})
print(f"Original text:\n  {TEXT}")
print(f"\nStep 1 output (summary):\n  {result['summary']}")
print(f"\nStep 2 output (title):\n  {result['title']}")
