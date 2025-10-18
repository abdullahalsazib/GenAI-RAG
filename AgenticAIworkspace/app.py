import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import MessagesState
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

# Load environment variables
os.environ["LANGSMITH_PROJECT"] = "AgenticAIworkspace"
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

# Initialize the LLM
llm = ChatGroq(api_key=groq_key, model="llama-3.1-8b-instant")

# Define a simple tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]
llm_with_tools = llm.bind_tools(tools)

sys_message = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Define State
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Node
def assistant(state: State):
    return {"messages": [llm_with_tools.invoke([sys_message] + state["messages"])]}
def make_default_graph():
    # Build graph
    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant") # <- back to the brain and make one more decition
    builder.add_edge("assistant", END)  # Terminate graph

    graph = builder.compile()
    return graph


graph=make_default_graph()
