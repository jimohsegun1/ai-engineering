"""LCEL chain composition: router chaining.

RunnableBranch sends an input down one of several chains depending on a
condition, instead of always running the same prompt. The condition here is
a simple keyword rule for reliability; a production router often replaces it
with an LLM classification step instead.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_huggingface import HuggingFacePipeline

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


CHAT_MODEL = "google/flan-t5-base"
QUESTIONS = [
    "What is 12 plus 15?",
    "What is the capital of Japan?",
]

print_step(1, "Build a chain per category")
llm = HuggingFacePipeline.from_model_id(
    model_id=CHAT_MODEL,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
)
math_prompt = PromptTemplate.from_template("Solve this math problem, answer only:\n\n{question}")
general_prompt = PromptTemplate.from_template("Answer this question concisely:\n\n{question}")

math_chain = math_prompt | llm | StrOutputParser()
general_chain = general_prompt | llm | StrOutputParser()


def is_math_question(inputs: dict) -> bool:
    keywords = ("plus", "minus", "times", "divided", "+", "-", "*", "/")
    return any(word in inputs["question"].lower() for word in keywords)


print_step(2, "Route each question with RunnableBranch")
router = RunnableBranch(
    (is_math_question, math_chain),
    general_chain,  # default branch
)

for question in QUESTIONS:
    route = "math_chain" if is_math_question({"question": question}) else "general_chain"
    answer = router.invoke({"question": question})
    print(f"\nQuestion: {question}")
    print(f"Routed to: {route}")
    print(f"Answer: {answer}")
