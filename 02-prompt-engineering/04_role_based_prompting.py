"""Prompt engineering: role-based prompting.

Assigning the model a persona shapes tone and style without changing the
underlying task — the same question is answered twice here, once per
persona, so you can compare how each one changes the response.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


CHAT_MODEL = "google/flan-t5-base"
QUESTION = "Why should I back up my files regularly?"
PERSONAS = ["a pirate", "a strict IT security officer"]

print_step(1, "Build a role-based prompt template")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
)
prompt = PromptTemplate.from_template(
    "You are {persona}. Answer the following question in that voice: {question}"
)
chain = prompt | llm | StrOutputParser()

print_step(2, "Ask the same question with two different personas")
for persona in PERSONAS:
    answer = chain.invoke({"persona": persona, "question": QUESTION})
    print(f"\nPersona: {persona}")
    print(f"Answer: {answer}")
