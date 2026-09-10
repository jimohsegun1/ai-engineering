"""LangGraph: the prebuilt tool-calling agent.

06-agents/03_tool_calling_agent.py builds a tool-calling agent by hand:
a prompt, create_tool_calling_agent, and AgentExecutor. LangGraph's
create_react_agent (from langgraph.prebuilt) does the same job — a loop of
"call the model, call a tool if requested, repeat until a final answer" —
as a single prebuilt, compiled graph. Same free local Ollama model as
06-agents.
"""

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

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


print_step(1, "Build the agent (one call, no prompt/executor boilerplate)")
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
agent = create_react_agent(llm, tools=[word_count])

print_step(2, "Run the agent")
result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})
final_message = result["messages"][-1]
print(f"Question: {QUESTION}")
print(f"Final answer: {final_message.content}")
