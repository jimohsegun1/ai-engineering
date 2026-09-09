"""Agents: the ReAct pattern, on Qorebit instead of Ollama.

Same ReAct setup as 01_react_agent.py, but with Qorebit's hosted model
(gpt-4o) as the LLM instead of local llama3.2:3b — this is a live Qorebit
call, unlike the rest of this project's Qorebit usage, since an agent demo
has no free partial run to fall back to; the whole point is invoking the
LLM.

Unlike llama3.2:3b (which loops forever and never reaches a final answer),
gpt-4o reliably gets to the correct answer — but not on the first try. It
tends to write out the whole Thought/Action/Observation/Final Answer
sequence in one go, predicting the tool's result instead of waiting for it,
which `handle_parsing_errors=True` catches and retries a few times until the
model settles into the correct one-step-at-a-time format and finishes
cleanly.
"""

import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


QOREBIT_BASE_URL = "https://api.qorebit.ai/v1"
CHAT_MODEL = "openai/gpt-4o"
QUESTION = "What is 23 times 7, plus 15?"

REACT_PROMPT = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
)


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '23 * 7 + 15'."""
    import numexpr

    try:
        return str(numexpr.evaluate(expression.strip("'\" ")).item())
    except Exception as e:
        return f"Error: could not evaluate {expression!r} ({e})"


print_step(1, "Build the ReAct agent (Qorebit LLM + tools + prompt)")
llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0,
    api_key=os.environ["QOREBIT_API_KEY"],
    base_url=QOREBIT_BASE_URL,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "AI Engineering RAG Learning App",
        "User-Agent": "rag-app-learning-project/1.0",
    },
)
tools = [calculator]
agent = create_react_agent(llm, tools, REACT_PROMPT)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

print_step(2, "Run the agent — watch it reason, act, and observe")
result = agent_executor.invoke({"input": QUESTION})
print(f"\nQuestion: {QUESTION}")
print(f"Final answer: {result['output']}")
