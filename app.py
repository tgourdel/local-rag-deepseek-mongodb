import os
import tempfile
import time
import streamlit as st
from chatpdf import RAG

st.set_page_config(page_title="Local RAG with DeepSeek and MongoDB")


def display_messages():
    """Display the chat history using Streamlit's native chat interface."""
    st.subheader("Chat History")
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    """Process the user input and generate an assistant response."""
    user_input = st.session_state.get("user_input", "").strip()
    if user_input:
        # Add user message to chat history
        st.session_state["messages"].append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Prepare conversation history for context (excluding the latest user message if desired)
        conversation_history = [
            msg["content"] for msg in st.session_state["messages"] if msg["role"] != "system"
        ]
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Generate the assistant response with context
                    agent_text = st.session_state["assistant"].ask(
                        user_input,
                        conversation_history=conversation_history,
                        k=st.session_state["retrieval_k"],
                        score_threshold=st.session_state["retrieval_threshold"],
                    )
                except ValueError as e:
                    agent_text = str(e)
            
            st.markdown(agent_text)
        
        # Add assistant response to chat history
        st.session_state["messages"].append({"role": "assistant", "content": agent_text})
        
        # Clear the input box
        st.session_state["user_input"] = ""


def upload_file():
    """Handle file upload and ingestion."""
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}..."):
            t0 = time.time()
            st.session_state["assistant"].ingestion(file_path)
            t1 = time.time()

        st.session_state["messages"].append(
            {"role": "system", "content": f"Ingested {file.name} in {t1 - t0:.2f} seconds"}
        )
        os.remove(file_path)


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = RAG()
    if "ingestion_spinner" not in st.session_state:
        st.session_state["ingestion_spinner"] = st.empty()
    if "retrieval_k" not in st.session_state:
        st.session_state["retrieval_k"] = 5
    if "retrieval_threshold" not in st.session_state:
        st.session_state["retrieval_threshold"] = 0.2
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""


def main():
    """Main app page layout."""
    initialize_session_state()

    st.header("RAG with Local DeepSeek R1")

    st.subheader("Upload a Document")
    st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="file_uploader",
        on_change=upload_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    # Retrieval settings
    st.subheader("Settings")
    st.session_state["retrieval_k"] = st.slider(
        "Number of Retrieved Results (k)", min_value=1, max_value=10, value=st.session_state["retrieval_k"]
    )
    st.session_state["retrieval_threshold"] = st.slider(
        "Similarity Score Threshold", min_value=0.0, max_value=1.0, value=st.session_state["retrieval_threshold"], step=0.05
    )

    # Display messages and text input
    display_messages()
    
    # Accept user input using the new chat input
    prompt = st.chat_input("Type your message here...")
    if prompt:
        st.session_state["user_input"] = prompt
        process_input()

    # Clear chat
    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        st.session_state["assistant"].clear()
        st.session_state["user_input"] = ""


if __name__ == "__main__":
    main()
