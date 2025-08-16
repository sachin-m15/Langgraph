import streamlit as st
from langchain_core.messages import HumanMessage
from backend.langgrapg_database_backend import chatbot, retrieve_all_threads
import uuid
import json

# **************************************** Utility Functions *************************


def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    return chatbot.get_state(config={"configurable": {"thread_id": thread_id}}).values[
        "messages"
    ]


# **************************************** Session Setup *****************************

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# **************************************** Sidebar UI ********************************

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})

        st.session_state["message_history"] = temp_messages

# **************************************** Main UI ***********************************

st.header("How can I help you?")

# Load conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here...")

if user_input:
    # Add user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    # Stream AI message & collect clean text
    final_response = ""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages",
        ):
            if hasattr(message_chunk, "content"):
                chunk_text = message_chunk.content

                # Try parsing JSON results safely
                try:
                    parsed = json.loads(chunk_text)
                    if (
                        isinstance(parsed, dict)
                        and "results" in parsed
                        and isinstance(parsed["results"], list)
                    ):
                        results_list = []
                        for r in parsed["results"]:
                            title = r.get("title", "Untitled")
                            snippet = r.get("snippet", "")
                            link = r.get("link", "")

                            if not snippet and "description" in r:
                                snippet = r["description"]
                            if not snippet:
                                snippet = "No summary available."

                            if link:
                                results_list.append(
                                    f"**[{title}]({link})**\n> {snippet}"
                                )
                            else:
                                results_list.append(f"**{title}**\n> {snippet}")

                        chunk_text = "\n\n".join(results_list)
                except (json.JSONDecodeError, TypeError):
                    pass

                # Update display
                final_response += chunk_text
                message_placeholder.markdown(final_response)

    # Save only clean text in history
    st.session_state["message_history"].append(
        {"role": "assistant", "content": final_response}
    )
