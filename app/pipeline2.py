import tempfile
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from app.Loader import load_document
from app.MetaData import Document
from app.Processing import split_documents, get_embeddings, create_vectorstore

from app.GroqClient import GroqClient
from app.Tools import create_document_tools


class SmartDocAssistant:

    def __init__(self):
        self.documents = {}
        self.vectorstores = {}
        self.model = GroqClient()
        self.tools = create_document_tools(self)
        self.agent, _ = self.model.build_agent()

    def ensure_agent(self):
        if self.agent is None:
            self.tools = create_document_tools(self)
            self.agent, _ = self.model.build_agent(self.tools)
        return self.agent

    # ---------------- LOAD DOCUMENT ----------------

    def load_document_file(self, uploaded_file):
        file_type = uploaded_file.name.split(".")[-1]

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        raw_docs = load_document(tmp_path, file_type)

        docs = []
        for i, doc in enumerate(raw_docs):
            metadata = {
                "filename": uploaded_file.name,
                "chunk_index": i,
                "uploaded_at": datetime.now(timezone.utc).isoformat()
            }
            docs.append(Document(page_content=doc.page_content, metadata=metadata))

        self.documents[uploaded_file.name] = docs
        return docs

    # ---------------- PROCESS ----------------

    def process_document(self, filename):
        chunks = split_documents(self.documents[filename])

        embeddings = get_embeddings()
        vectorstore = create_vectorstore(chunks, embeddings)

        self.vectorstores[filename] = vectorstore
        return "Processed"

    # ---------------- RETRIEVE ----------------

    def retrieve(self, query, filename):
        vs = self.vectorstores.get(filename)
        if not vs:
            return ""

        results = vs.similarity_search(query, k=3)
        return "\n\n".join([r.page_content for r in results])

    # ---------------- SUMMARIZE ----------------

    def summarize_document(self, filename):
        chunks = list(self.vectorstores[filename].docstore._dict.values())

        def summarize(c):
            return self.agent.invoke({
                "messages": [{
                    "role": "user",
                    "content": f"Summarize:\n{c.page_content}"
                }]
            })["messages"][-1].content

        with ThreadPoolExecutor(max_workers=5) as ex:
            summaries = list(ex.map(summarize, chunks[:10]))

        return "\n\n".join(summaries)

    def summarize_chapter(self, filename, chapter):
        return f"(Demo) Summary of {chapter} from {filename}"

    # ---------------- MAIN ENTRY ----------------

    def ask(self, question, filename=None):
        if filename and filename not in self.documents:
            return "Document not found."

        agent = self.ensure_agent()

        response = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": f"""
                    User is working with file: {filename}

                    Question:
                    {question}
                    """
                }
            ]
        })

        return response["messages"][-1].content