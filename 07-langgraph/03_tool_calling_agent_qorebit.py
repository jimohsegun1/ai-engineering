"""LangGraph: the prebuilt tool-calling agent, on Qorebit instead of Ollama.

Same setup as 03_tool_calling_agent.py, but with Qorebit's hosted model
(gpt-4o) as the LLM. This is a live Qorebit call — see the note in
06-agents/01_react_agent_qorebit.py about why agent demos don't have a free
partial run to fall back to.

Needs `disable_streaming=True` on the LLM, same reason as
06-agents/03_tool_calling_agent_qorebit.py: LangGraph's prebuilt agent
streams the model internally to plan each step, and Qorebit's gateway
mangles the tool_call id field when a tool-calling response is streamed.
"""

import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

RULE = "=" * 70


def print_step(number: int, title: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}: {title}")
    print(RULE)


QOREBIT_BASE_URL = "https://api.qorebit.ai/v1"
CHAT_MODEL = "openai/gpt-4o"
QUESTION = "How many words are in the sentence 'the quick brown fox jumps over the lazy dog'?"


@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())


print_step(1, "Build the agent (Qorebit LLM, one call, no prompt/executor boilerplate)")
llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0,
    disable_streaming=True,  # Qorebit mangles tool_call ids when streamed, see module docstring
    api_key=os.environ["QOREBIT_API_KEY"],
    base_url=QOREBIT_BASE_URL,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "AI Engineering RAG Learning App",
        "User-Agent": "rag-app-learning-project/1.0",
    },
)
agent = create_react_agent(llm, tools=[word_count])

print_step(2, "Run the agent")
result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})
final_message = result["messages"][-1]
print(f"Question: {QUESTION}")
print(f"Final answer: {final_message.content}")
