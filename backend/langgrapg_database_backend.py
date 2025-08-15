from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_tavily import TavilySearch

# Used for real time informations
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph.message import add_messages
import sqlite3

# ******************************LLM model**************************
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="llama3-8b-8192")



# *******************************Adding tool for real time news**********************************

tool = TavilySearch(
    max_result=2, tavily_api_key="Your API key"
)
# Binding multiple tools together
tools = [tool]

# Binding the tools with llm model
tool_model = model.bind_tools(tools)


# Define the ChatState
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Define a single chat_node
def chat_node(state: ChatState):
    messages = state["messages"]
    messages = messages[-5:]
    response = tool_model.invoke(messages)
    return {"messages": [response]}


# Create a StateGraph
graph = StateGraph(ChatState)
graph.add_node("chatbot", chat_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", tools_condition)
graph.add_edge("chatbot", END)

# ********************************making database********************
con = sqlite3.connect(database="chatbot.db", check_same_thread=False)

# Initialize SQLite checkpointer
checkpointer = SqliteSaver(conn=con)

# Compile the graph
chatbot = graph.compile(checkpointer=checkpointer)


# Function to retrieve all threads from database
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
