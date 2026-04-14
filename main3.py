import streamlit as st
from app.pipeline3 import SmartDocAssistant
import time
import hashlib

st.set_page_config(page_title="Document Summerizer Agent", layout="wide")
st.title("Document Summerizer Agent")

#intialize variables
if "agent" not in st.session_state:
    st.session_state.agent = SmartDocAssistant()

if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = set()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = {}


#clear chat button
col1, col2 = st.columns([4, 1])

with col2:
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.processed_docs = set()
        st.session_state.agent.memory.clear()
        st.success("Chat cleared!")

st.divider()

#File upload
uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "docx", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Loading & Processing documents..."):
        for f in uploaded_files:

            file_bytes = f.getvalue()
            file_id = hashlib.md5(file_bytes).hexdigest()

            if file_id not in st.session_state.processed_docs:
                st.session_state.agent.load_document_file(f)
                st.session_state.agent.process_document(f.name)

                st.session_state.processed_docs.add(file_id)
                st.session_state.documents[f.name] = True

    st.success("Documents ready!")


#Select document if you have multiple
filename = None

if st.session_state.documents:
    filename = st.selectbox(
        "Select document",
        options=list(st.session_state.documents.keys())
    )

st.divider()


#Streaming output
def stream_text(text):
    placeholder = st.empty()

    for i in range(1, len(text) + 1):
        placeholder.markdown(text[:i])
        time.sleep(0.005)


#chat history
st.subheader("Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("Ask anything..."):

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):

            if not filename:
                response = "Please upload and select a document."

            elif filename not in st.session_state.documents:
                response = "Document not processed."

            else:
                response = st.session_state.agent.ask_with_sources(
                    prompt,
                    filename
                )[0]  # only take response, ignore sources

        stream_text(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })