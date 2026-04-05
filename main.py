import streamlit as st
from app.AgentPipeline import SmartDocAssistant

st.set_page_config(page_title="Multi Document Summarizer Agent", layout="wide")
st.title("Document Summerizer Agent")


# Prepare agent Pipeline object
if "MainAgent" not in st.session_state:
    st.session_state.MainAgent = SmartDocAssistant()

# Track processed documents
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = set()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query_text" not in st.session_state:
    st.session_state.query_text = ""

#file uploader
uploaded_files = st.file_uploader(
    "Upload PDF / TXT / DOCX / CSV",
    type=["pdf", "txt", "docx", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Loading files..."):
        for f in uploaded_files:
            st.session_state.MainAgent.load_document_file(f)
    st.success("✅ Files loaded!")


if st.session_state.MainAgent.documents:

    filename = st.selectbox(
        "Select document",
        options=list(st.session_state.MainAgent.documents.keys())
    )

    # -------------------------
    # STEP 1: PROCESS
    # -------------------------
    if st.button("⚙️ Process Document"):
        with st.spinner("Processing document..."):
            st.session_state.MainAgent.process_document(filename)
        st.session_state.processed_docs.add(filename)
        st.success("Document processed!")

    # -------------------------
    # STEP 2: SUMMARIZE (ONLY AFTER PROCESS)
    # -------------------------
    if filename in st.session_state.processed_docs:

        if st.button("📝 Summarize Document"):
            with st.spinner("Generating summary..."):
                summary = st.session_state.MainAgent.summarize_document(filename)

            st.subheader(f"📌 Summary of {filename}")
            st.write(summary)

    else:
        st.info("⚠️ Please process the document first.")


# Initialize session state for answer
if "answer" not in st.session_state:
    st.session_state.answer = ""

# Form to handle query input
with st.form(key="query_form", clear_on_submit=True):
    query = st.text_input("Ask a question", key="query_text")
    submit_button = st.form_submit_button("Send")

    if submit_button and query:
        with st.spinner("Agent thinking..."):
            if 'filename' in locals() and filename in st.session_state.processed_docs:
                st.session_state.answer = st.session_state.MainAgent.ask(query, filename)
            else:
                st.session_state.answer = st.session_state.MainAgent.ask(query)

# Display the answer
if st.session_state.answer:
    st.subheader("💬 Answer")
    st.write(st.session_state.answer)