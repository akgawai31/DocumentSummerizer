from typing import Optional
from langchain_core.tools import tool


def create_document_tools(context):

    assistant = context.assistant

    @tool
    def search_document(query: str, filename: Optional[str] = None) -> str:
        """
        Search across documents.
        If filename is provided → search only that file.
        If not → search all documents.
        """

        if not assistant.vectorstores:
            return "No documents have been processed yet."

        results = []

        # CASE 1: filename not provided → search all
        if not filename:
            for fname, vs in assistant.vectorstores.items():
                docs = vs.similarity_search(query, k=2)
                for d in docs:
                    results.append(f"[{fname}]\n{d.page_content}")

        # CASE 2: filename provided → search only that file
        else:
            vs = assistant.vectorstores.get(filename)

            if not vs:
                return f"Document '{filename}' not found."

            docs = vs.similarity_search(query, k=3)
            for d in docs:
                results.append(f"[{filename}]\n{d.page_content}")

        return "\n\n".join(results) if results else "No results found."

    # keep your other tools as-is

    @tool
    def summarize_document(filename: str) -> str:
        """Summarize the full document given its filename."""
        return context.assistant.summarize_document(filename)


    @tool
    def summarize_chapter(filename: str, chapter: str) -> str:
        """Summarize a specific chapter from a document."""
        return context.assistant.summarize_chapter(filename, chapter)
    

    return [search_document, summarize_document, summarize_chapter]