import streamlit as st
from app2.pipeline import SmartDocAssistant
import hashlib
import time

st.set_page_config(page_title="Document Summarizer", layout="wide")
st.title("Document Summarizer RAG - Agent")

#Initialize 
if "agent" not in st.session_state:
    st.session_state.agent = SmartDocAssistant()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = set()

if "documents" not in st.session_state:
    st.session_state.documents = {}


#Buttons
col1, col2 = st.columns(2)

#Only Clears chat history
with col1:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

#clear everything including document
with col2:
    if st.button("Reset Everything"):
        st.session_state.clear()
        st.rerun()

st.divider()


#FIle upload
uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "docx", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:

        file_id = hashlib.md5(f.getvalue()).hexdigest()

        if file_id not in st.session_state.processed_docs:
            with st.spinner(f"Processing {f.name}..."):
                st.session_state.agent.load_document_file(f)
                st.session_state.agent.process_document(f.name)

                st.session_state.processed_docs.add(file_id)
                st.session_state.documents[f.name] = True

    st.success("Documents ready!")

#Select Document
filename = None

if st.session_state.documents:
    filename = st.selectbox(
        "Select document",
        list(st.session_state.documents.keys())
    )

    st.session_state.agent.current_file = filename

st.divider()


#Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


#Thinking animation
def show_thinking(placeholder, duration=1.2):
    steps = ["Thinking", "Thinking.", "Thinking..", "Thinking..."]
    start = time.time()

    while time.time() - start < duration:
        for step in steps:
            placeholder.markdown(step)
            time.sleep(0.3)


#Streaming text
def stream_text(text, placeholder, speed=0.03):
    words = text.split(" ")
    streamed = ""

    for word in words:
        streamed += word + " "
        placeholder.markdown(streamed + "▌")
        time.sleep(speed)

    placeholder.markdown(streamed)
    return streamed


#Input
if prompt := st.chat_input("Ask something...", disabled=not filename):

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()

        try:
            show_thinking(placeholder, duration=1.5)
            full_response = st.session_state.agent.ask_sync(prompt)

            if not full_response:
                full_response = "No response received."

            response = stream_text(full_response, placeholder, speed=0.03)

        except Exception as e:
            response = f"Error: {str(e)}"
            placeholder.markdown(response)

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })