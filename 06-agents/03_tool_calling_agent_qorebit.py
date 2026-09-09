"""Agents: native tool-calling, on Qorebit instead of Ollama.

Same tool-calling setup as 03_tool_calling_agent.py, but with Qorebit's
hosted model (gpt-4o) as the LLM. This is a live Qorebit call — see the note
in 01_react_agent_qorebit.py about why agent demos don't have a free partial
run to fall back to.

Needs `disable_streaming=True` on the LLM: AgentExecutor streams the model's
response internally to plan each step, and Qorebit's gateway mangles the
tool_call id field when a tool-calling response is streamed, sending back an
id hundreds of characters long instead of a normal one — the API then
rejects the next request with a 400 ("string too long"). Disabling
streaming makes the client request (and get back) the response in one
piece instead, which avoids the bug entirely.
"""

import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
QUESTION = "How many words are in the sentence 'the quick brown fox jumps over the lazy dog'?"


@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())


print_step(1, "Build the tool-calling agent (Qorebit LLM + tools + prompt)")
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
