"""LCEL chain composition: parallel chaining.

RunnableParallel runs multiple independent chains against the same input at
once and collects their outputs into a single dict, instead of running them
one after another.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_huggingface import HuggingFacePipeline

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


CHAT_MODEL = "google/flan-t5-base"
TEXT = (
    "Prompt engineering is the practice of designing the wording, structure, and "
    "examples in a prompt so a language model reliably produces the output you want."
)

print_step(1, "Build two independent chains over the same input")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
)
summary_prompt = PromptTemplate.from_template("Summarize this text in one sentence:\n\n{text}")
keyword_prompt = PromptTemplate.from_template("List 3 keywords from this text, comma-separated:\n\n{text}")

summary_chain = summary_prompt | llm | StrOutputParser()
keyword_chain = keyword_prompt | llm | StrOutputParser()

print_step(2, "Run both in parallel with RunnableParallel")
parallel_chain = RunnableParallel(summary=summary_chain, keywords=keyword_chain)
result = parallel_chain.invoke({"text": TEXT})

print(f"Input text:\n  {TEXT}")
print(f"\nsummary:  {result['summary']}")
print(f"keywords: {result['keywords']}")
