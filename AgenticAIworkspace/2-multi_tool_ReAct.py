import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import MessagesState

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
# Bind the tool to the LLM
llm_with_tools = llm.bind_tools(tools)

sys_message = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")


# Node
def assistent(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_message] + state["messages"])]}

# Builder
builder = StateGraph(MessagesState)

builder.add_node("assistent", assistent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistent")
builder.add_conditional_edges(
    "assistent",
    tools_condition
)

builder.add_edge("tools", "assistent")
graph = builder.compile()
messages = HumanMessage(content="Add 3 and 4. Multiply the output by 2. Divide the output by 5")

messages = graph.invoke({"messages": messages})

print(messages)