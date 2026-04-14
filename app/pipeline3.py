# app/SmartDocAssistant.py

from app.GroqAgent import GroqClient
from app.Loader import load_document
from app.MetaData import Document
from app.Processing import split_documents, get_embeddings, create_vectorstore
from app.Memory import ConversationMemory
from typing import Optional

from app.tool_context import ToolContext
from app.toolCall import create_document_tools

import tempfile
import re
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed


SYSTEM_PROMPT = """
You are an intelligent document agent.

You can:
- Answer questions using document context
- Summarize full documents
- Summarize chapters

Rules:
- Use tools only when needed
- Do NOT call tools repeatedly
- Call a tool at most once per query
- After using a tool → generate final answer

- For Q&A → use search_document
- For chapter questions → use summarize_chapter
- For full document → use summarize_document

Be clear, structured, and concise.
"""


class SmartDocAssistant:

    def __init__(self):
        self.documents = {}
        self.vectorstores = {}
        self.chapter_map = {}
        self.memory = ConversationMemory()

        self.model = GroqClient()

        # ✅ RAW LLM (no tools)
        self.llm = self.model.get_llm()

        self.context = ToolContext(self)
        self.tools = create_document_tools(self.context)

        # ✅ limit iterations → prevents loops
        self.agent = self.model.build_agent(
            self.tools
        )

    # ---------------------------
    # Document Loading
    # ---------------------------
    def load_document_file(self, uploaded_file):

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        raw_docs = load_document(
            tmp_path,
            uploaded_file.name.split(".")[-1]
        )

        docs = []
        for i, doc in enumerate(raw_docs):
            docs.append(Document(
                page_content=doc.page_content,
                metadata={
                    "filename": uploaded_file.name,
                    "chunk_index": i,
                    "uploaded_at": datetime.now(timezone.utc).isoformat()
                }
            ))

        self.documents[uploaded_file.name] = docs
        self._detect_sections(uploaded_file.name)


    def _to_text(self, x):
        if hasattr(x, "content"):
            return x.content
        return str(x)

    # ---------------------------
    # Processing
    # ---------------------------
    def process_document(self, filename):

        chunks = split_documents(self.documents[filename])

        embeddings = get_embeddings()
        self.vectorstores[filename] = create_vectorstore(chunks, embeddings)

    # ---------------------------
    # Section Detection
    # ---------------------------
    def _detect_sections(self, filename):

        self.chapter_map[filename] = {}

        current = "introduction"
        self.chapter_map[filename][current] = []

        pattern = re.compile(
            r"(chapter|ch|unit|section)\s*\d+[:.\-\s]*([^\n]*)",
            re.I
        )

        for i, doc in enumerate(self.documents[filename]):

            match = pattern.search(doc.page_content)

            if match:
                current = match.group().lower()
                self.chapter_map[filename][current] = []

            self.chapter_map[filename][current].append(i)

    # ---------------------------
    # SAFE LLM CALL
    # ---------------------------
    def _llm_call(self, prompt):
        try:
            result = self.llm.invoke(prompt)
            return self._to_text(result)
        except Exception as e:
            return f"Error generating response: {str(e)}"
    # ---------------------------
    # FULL DOCUMENT SUMMARY (SAFE THREADPOOL)
    # ---------------------------
    def summarize_document(self, filename):

        if filename not in self.vectorstores:
            return "Document not processed."

        chunks = list(self.vectorstores[filename].docstore._dict.values())

        def summarize_chunk(chunk):
            return self._llm_call(
                f"Summarize clearly and concisely:\n\n{chunk.page_content}"
            )

        summaries = []

        # ✅ SAFE THREADPOOL (limited workers)
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(summarize_chunk, c) for c in chunks]

            for future in as_completed(futures):
                summaries.append(self._to_text(future.result()))

        # ---------------------------
        # Reduce Phase (SEQUENTIAL)
        # ---------------------------
        while len(summaries) > 1:

            new_summaries = []

            for i in range(0, len(summaries), 10):

                batch = summaries[i:i+10]

                combined = self._llm_call(
                    "Combine into structured summary:\n\n" +
                    "\n\n".join(self._to_text(b) for b in batch)
                )

                new_summaries.append(combined)

            summaries = new_summaries

        return summaries[0]

    # ---------------------------
    # CHAPTER SUMMARY
    # ---------------------------
    def summarize_chapter(self, filename, chapter):

        if filename not in self.chapter_map:
            return "No chapter data found."

        idxs = self.chapter_map[filename].get(chapter.lower())

        if not idxs:
            return "Chapter not found."

        text = "\n\n".join(
            self.documents[filename][i].page_content
            for i in idxs
        )

        return self._llm_call(
            f"""
            Summarize this chapter in structured format:

            Chapter: {chapter}

            Content:
            {text}
            """
        )

    # ---------------------------
    # AGENT EXECUTION
    # ---------------------------
    def run_agent(self, question, filename):

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.memory.get(),
            {
                "role": "user",
                "content": f"Document: {filename}\n\nQuestion: {question}"
            }
        ]

        response = self.agent.invoke({
            "messages": messages
        })

        # 🔥 SAFETY CHECK: prevent infinite tool loops
        if len(response["messages"]) > 20:
            return "Stopped due to excessive tool calls."

        answer = response["messages"][-1].content

        self.memory.add("user", question)
        self.memory.add("assistant", answer)

        return answer

    # ---------------------------
    # PUBLIC API
    # ---------------------------
    def ask(self, question, filename):

        if filename not in self.documents:
            return "Document not loaded."

        return self.run_agent(question, filename)