from app.GroqAgent import GroqClient
from app.Loader import load_document
import textwrap,tempfile,re
from datetime import datetime, timezone
from app.MetaData import Document
from app.Processing import split_documents, get_embeddings, create_vectorstore
from app.Tools import create_document_tools
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed

class SmartDocAssistant:

    def __init__(self):
        self.documents = {}
        self.vectorstores = {}
        self.chunks = {} 
        self.chapter_map = {}
        self.model = GroqClient()
        self.tools = create_document_tools(self)
        self.agent, _ = self.model.build_agent()
        print("Agent ready at initialization:", self.agent)

    def ensure_agent(self):
        if self.agent is None:
            self.model = GroqClient()
            self.tools = create_document_tools(self)
            self.agent, _ = self.model.build_agent()
        return self.agent
    

    def _detect_sections(self, filename):
        self.chapter_map[filename] = {}
        chunks = self.documents.get(filename, [])
        current_section = "introduction"
        self.chapter_map[filename][current_section] = []

        section_pattern = re.compile(
            r"^(chapter|ch|unit|lesson|module|section|part)\s*\d+[\.: -]?\s*.*",
            re.IGNORECASE
        )

        for idx, chunk in enumerate(chunks):
            lines = chunk.page_content.splitlines()
            for line in lines:
                line_clean = line.strip()
                if section_pattern.match(line_clean):
                    match = re.search(r"(chapter|ch|unit|lesson|module|section|part)\s*(\d+)", line_clean, re.IGNORECASE)
                    if match:
                        keyword = match.group(1).lower()
                        number = match.group(2)
                        if keyword == "ch":
                            keyword = "chapter"
                        current_section = f"{keyword} {number}".lower()
                        self.chapter_map[filename][current_section] = []
                        break
            self.chapter_map[filename][current_section].append(idx)

    
    def extract_section_from_query(self, query):
        match = re.search(r"(chapter|ch|unit|lesson|module|section|part)\s*(\d+)", query, re.IGNORECASE)
        if match:
            keyword = match.group(1).lower()
            number = match.group(2)
            if keyword == "ch":
                keyword = "chapter"
            return f"{keyword} {number}".lower()
        return None
    

    def detect_intent(self, query):
        query_lower = query.lower().strip()

        section_pattern = r"(chapter|ch|unit|lesson|module|section|part)[\s\-]*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        if re.search(section_pattern, query_lower):
            return "section_summary"

        full_doc_keywords = [
            "summarize document",
            "summary of document",
            "summarize all",
            "full summary",
            "entire document"
        ]
        if any(kw in query_lower for kw in full_doc_keywords):
            return "document_summary"

        if "summarize" in query_lower or "summary" in query_lower:
            return "document_summary"

        return "qa"



    def load_document_file(self, uploaded_file):
        file_type = uploaded_file.name.split(".")[-1]

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        raw_docs = load_document(tmp_path, file_type)

        docs_with_metadata = []
        for i, doc in enumerate(raw_docs):
            metadata = doc.metadata.copy() if hasattr(doc, "metadata") else {}
            metadata.update({
                "filename": uploaded_file.name,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "chunk_index": i
            })
            docs_with_metadata.append(Document(page_content=doc.page_content, metadata=metadata))

        self.documents[uploaded_file.name] = docs_with_metadata
        # Build chapter map
        self._detect_sections(uploaded_file.name)
        return docs_with_metadata
    
    def process_document(self, filename):
        if filename not in self.documents:
            raise ValueError(f"No document named {filename} uploaded.")

        chunks = split_documents(self.documents[filename])
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "filename": filename,
                "chunk_index": i,
                "uploaded_at": datetime.now(timezone.utc).isoformat()
            })

        embeddings = get_embeddings()
        vectorstore = create_vectorstore(chunks, embeddings)
        self.vectorstores[filename] = vectorstore

        return "Document Processed Successfully"
    
    def retrieve(self, query, filename):
        vectorstore = self.vectorstores.get(filename)
        if not vectorstore:
            return f"Document '{filename}' not processed."
        # Return top 3 relevant chunks
        results = vectorstore.similarity_search(query, k=3)
        return "\n\n".join([r.page_content for r in results])
    
    

    def summarize_document(self, filename, max_chunks=15, max_workers=5):
        if filename not in self.vectorstores:
            return "Document not processed yet."

        # Get all chunks from the vectorstore
        chunks = list(self.vectorstores[filename].docstore._dict.values())
        if not chunks:
            return "No content to summarize."

        def summarize_chunk(c):
            message = {
                "role": "user",
                "content": (
                    "Provide a detailed summary of the following text. "
                    "Include key points, important facts, and explanations:\n\n"
                    f"{c.page_content}"
                )
            }
            response = self.ensure_agent().invoke({"messages": [message]})
            return response["messages"][-1].content

        # Parallel summarization
        chunk_summaries = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(summarize_chunk, c) for c in chunks[:max_chunks]]
            for future in as_completed(futures):
                chunk_summaries.append(future.result())

        # Combine summaries into a detailed final summary
        combined_text = "\n\n".join(chunk_summaries)

        final_message = {
            "role": "user",
            "content": (
                "Create a comprehensive and well-structured summary from the following partial summaries.\n\n"
                "Requirements:\n"
                "- Preserve all key ideas\n"
                "- Expand where necessary for clarity\n"
                "- Organize into sections with headings\n"
                "- Use bullet points where helpful\n"
                "- Keep it detailed but readable\n\n"
                f"{combined_text}"
            )
        }

        final_response = self.ensure_agent().invoke({"messages": [final_message]})
        return final_response["messages"][-1].content
    

    def summarize_chapter(self, filename, chapter_name, max_workers=5):
        if filename not in self.documents:
            return f"Document '{filename}' not found."

        if filename not in self.chapter_map:
            return "Chapters not detected."

        chapter_chunks_idx = self.chapter_map[filename].get(chapter_name)

        if not chapter_chunks_idx:
            return f"{chapter_name} not found in document."

        chunks = [self.documents[filename][i] for i in chapter_chunks_idx]

        def summarize_chunk(chunk):
            message = {
                "role": "user",
                "content": (
                    "Summarize this part of a chapter in detail:\n\n"
                    f"{chunk.page_content}"
                )
            }
            response = self.ensure_agent().invoke({"messages": [message]})
            return response["messages"][-1].content

        summaries = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(summarize_chunk, c) for c in chunks]
            for future in as_completed(futures):
                summaries.append(future.result())

        combined = "\n\n".join(summaries)

        final_prompt = {
            "role": "user",
            "content": (
                f"Create a well-structured summary of {chapter_name}.\n\n"
                "- Keep it detailed\n"
                "- Use headings\n"
                "- Use bullet points\n\n"
                f"{combined}"
            )
        }

        final_response = self.ensure_agent().invoke({"messages": [final_prompt]})
        return final_response["messages"][-1].content
    


    def detect_intent(self, query):
        query_lower = query.lower().strip()

        # Match section-specific patterns: numeric or word numbers
        section_pattern = r"(chapter|ch|unit|lesson|module|section|part)[\s\-]*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        if re.search(section_pattern, query_lower):
            return "section_summary"

        # Full document summary keywords
        full_doc_keywords = [
            "summarize document",
            "summary of document",
            "summarize all",
            "full summary",
            "entire document"
        ]
        if any(kw in query_lower for kw in full_doc_keywords):
            return "document_summary"

        # Generic summary request
        if "summarize" in query_lower or "summary" in query_lower:
            return "document_summary"

        # Fallback to normal Q&A
        return "qa"



    def ask(self, question, filename=None):

        if not filename:
            return "Please provide a filename."

        if filename not in self.documents:
            return f"Document '{filename}' not found."

        intent = self.detect_intent(question)

        # Section summary
        if intent == "section_summary":
            section_name = self.extract_section_from_query(question)

            if not section_name:
                return "Couldn't detect section."

            return self.summarize_chapter(filename, section_name)

        # Full summary
        elif intent == "document_summary":
            return self.summarize_document(filename)

        # Q&A
        else:
            context_text = ""

            if filename in self.vectorstores:
                retrieved = self.retrieve(question, filename)
                if retrieved.strip():
                    context_text = f"{retrieved}\n\n"

            messages = [{
                "role": "user",
                "content": f"""
                You are a smart document assistant.

                Context:
                {context_text}

                Question:
                {question}
                """}]

            response = self.ensure_agent().invoke({"messages": messages})
            return response["messages"][-1].content
