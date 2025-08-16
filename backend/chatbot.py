from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Set HuggingFace API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load the HuggingFace model
endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)
model = ChatHuggingFace(llm=endpoint)


# Define the ChatState
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Define a node for the graph
def chat_node(state: ChatState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


# Create a StateGraph
graph = StateGraph(ChatState)
graph.add_node("chatbot", chat_node)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# Initialize in-memory checkpoint
memory = InMemorySaver()

# Compile the graph
chatbot = graph.compile(checkpointer=memory)
