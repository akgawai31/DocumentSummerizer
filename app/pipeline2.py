import hashlib
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from app.Loader import load_document
from app.MetaData import Document
from app.Processing import split_documents, get_embeddings, create_vectorstore
from app.GroqClient import GroqClient


# ---------------- SIMPLE CACHE ----------------

class Cache:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value


# ---------------- RERANKER ----------------

class SimpleReranker:
    """
    Lightweight reranker (no external model required)
    Improves relevance by keyword overlap scoring.
    """

    @staticmethod
    def score(query: str, text: str) -> float:
        q_tokens = set(query.lower().split())
        t_tokens = set(text.lower().split())

        if not q_tokens:
            return 0

        return len(q_tokens.intersection(t_tokens)) / len(q_tokens)

    def rerank(self, query: str, docs: List):
        scored = [
            (self.score(query, d.page_content), d)
            for d in docs
        ]

        scored.sort(reverse=True, key=lambda x: x[0])

        return [d for _, d in scored]


# ---------------- RAG CORE ----------------

class SmartDocAssistant:

    def __init__(self):
        self.docs: Dict[str, dict] = {}

        self.cache = Cache()
        self.reranker = SimpleReranker()

        self.embeddings = get_embeddings()
        self.llm = GroqClient()

    # ---------------- LOAD ----------------

    def load_document(self, uploaded_file):
        file_type = uploaded_file.name.split(".")[-1]

        raw_docs = load_document(uploaded_file, file_type)

        # simple chapter grouping (fallback safe)
        chapters = {"FULL_DOC": raw_docs}

        self.docs[uploaded_file.name] = {
            "chapters": chapters,
            "vectorstores": {}
        }

        return {"file": uploaded_file.name}

    # ---------------- PROCESS ----------------

    def process_document(self, filename: str):
        data = self.docs.get(filename)
        if not data:
            return "Document not found"

        for chapter, docs in data["chapters"].items():
            chunks = split_documents(docs)

            data["vectorstores"][chapter] = create_vectorstore(
                chunks,
                self.embeddings
            )

        return {"status": "processed"}

    # ---------------- RETRIEVAL ----------------

    def retrieve(self, query: str, filename: str, chapter: Optional[str] = None, k: int = 10):

        cache_key = hashlib.md5(f"{filename}:{chapter}:{query}".encode()).hexdigest()
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        data = self.docs.get(filename)
        if not data:
            return []

        results = []

        # 1. vector retrieval
        if chapter:
            vs = data["vectorstores"].get(chapter)
            if vs:
                results = vs.similarity_search(query, k=k)
        else:
            for vs in data["vectorstores"].values():
                results.extend(vs.similarity_search(query, k=3))

        # 2. rerank results
        results = self.reranker.rerank(query, results)

        # 3. limit final context
        results = results[:6]

        self.cache.set(cache_key, results)

        return results

    # ---------------- CONTEXT BUILDER ----------------

    def build_context(self, docs: List) -> str:
        seen = set()
        context_parts = []

        for d in docs:
            text = d.page_content.strip()

            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue

            seen.add(h)
            context_parts.append(text)

        return "\n\n".join(context_parts)

    # ---------------- ASK (CORE RAG) ----------------

    def ask(self, question: str, filename: str, chapter: Optional[str] = None):

        docs = self.retrieve(question, filename, chapter)

        context = self.build_context(docs)

        prompt = f"""
            You are a precise document assistant.

            Use ONLY the context below.

            Context:
            {context}

            Question:
            {question}

            Rules:
            - If answer is not in context, say "Not found in document"
            - Be concise
        """

        response = self.llm.invoke({
            "messages": [{
                "role": "user",
                "content": prompt
            }]
        })

        return response["messages"][-1].content

    # ---------------- SUMMARIZATION ----------------

    def summarize_document(self, filename: str):

        data = self.docs.get(filename)
        if not data:
            return "Not found"

        summaries = []

        for chapter, vs in data["vectorstores"].items():
            docs = vs.similarity_search("", k=8)

            context = self.build_context(docs)

            prompt = f"""
                Summarize this section clearly:

                {context}
            """

            res = self.llm.invoke({
                "messages": [{
                    "role": "user",
                    "content": prompt
                }]
            })

            summaries.append(f"### {chapter}\n{res}")

        return "\n\n".join(summaries)