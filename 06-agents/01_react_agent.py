"""Agents: the ReAct pattern.

Unlike a chain, an agent doesn't follow a fixed sequence of steps — the LLM
decides at each turn whether to call a tool or give a final answer, based on
its own reasoning. ReAct ("Reason + Act") is the classic text-based version
of this: the model writes a Thought, an Action (a tool name), and an Action
Input, the tool runs, and the Observation feeds back into the next Thought,
looping until it reaches a Final Answer. Runs on Ollama instead of the local
flan-t5-base used elsewhere in this project — a small text2text model can't
reliably produce this format, but a real instruction-tuned model like
llama3.2 can.

Even llama3.2:3b isn't fully reliable at this text-based format, though: it
sometimes computes the right answer but never actually writes the
"Final Answer:" line the parser is watching for, and just loops until
max_iterations cuts it off. That's a real, known brittleness of ReAct
prompting on smaller models — compare with 03_tool_calling_agent.py, which
uses the model's native tool-calling instead of text parsing and doesn't
have this problem.
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


OLLAMA_MODEL = "llama3.2:3b"
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


print_step(1, "Build the ReAct agent (LLM + tools + prompt)")
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
tools = [calculator]
agent = create_react_agent(llm, tools, REACT_PROMPT)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=5
)

print_step(2, "Run the agent — watch it reason, act, and observe")
result = agent_executor.invoke({"input": QUESTION})
print(f"\nQuestion: {QUESTION}")
print(f"Final answer: {result['output']}")
