"""Agents: the modern tool-calling agent.

01_react_agent.py gets the model to choose a tool by writing plain text in a
very specific format (Thought/Action/Action Input) that the agent then has
to parse — brittle, and wasted tokens. create_tool_calling_agent instead
relies on the model's own native function-calling ability (the same
mechanism used for structured tool APIs): the model returns a structured
tool call directly, no text parsing needed. Requires a model that supports
tool calling, which llama3.2 does.
"""

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


OLLAMA_MODEL = "llama3.2:3b"
QUESTION = "How many words are in the sentence 'the quick brown fox jumps over the lazy dog'?"


@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())


print_step(1, "Build the tool-calling agent (LLM + tools + prompt)")
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
tools = [word_count]
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use tools when they help answer the question."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print_step(2, "Run the agent — the model calls the tool directly, no text parsing")
result = agent_executor.invoke({"input": QUESTION})
print(f"\nQuestion: {QUESTION}")
print(f"Final answer: {result['output']}")
