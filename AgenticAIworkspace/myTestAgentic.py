import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables
os.environ["LANGSMITH_PROJECT"] = "AgenticAIworkspace"
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

# Initialize the LLM
llm = ChatGroq(api_key=groq_key, model="llama-3.1-8b-instant")

# Define a simple tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

# Bind the tool to the LLM
llm_with_tool = llm.bind_tools([add])

# Define the state for LangGraph
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define a node function that uses the LLM with the tool
def llm_tool(state: State):
    return {"messages": [llm_with_tool.invoke(state["messages"])]}

# Define available tools
tools = [add]

# Create a graph builder
builder = StateGraph(State)

# Add nodes
builder.add_node("llm_tool", llm_tool)
builder.add_node("tools", ToolNode(tools))

# Add edges
builder.add_edge(START, "llm_tool")
builder.add_conditional_edges("llm_tool", tools_condition)
builder.add_edge("tools", END)  # ✅ fixed typo: 'tools', not 'tool'
builder.add_edge("llm_tool", END)

# Compile the graph
graph = builder.compile()

# Run the graph with an initial message
input_messages = [HumanMessage(content="what is LLM?")]
result = graph.invoke({"messages": input_messages})

# Print all messages
# for msg in result["messages"]:
#     print(msg)

print(result)