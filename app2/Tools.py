from langchain.tools import tool

def create_document_tools(assistant):

    @tool
    def retrieve_document_context(query: str):
        """Retrieve relevant document chunks using vector search."""
        return assistant.retrieve(query)

    @tool
    def get_section_map():
        """Return document structure (units/sections map)."""
        filename = assistant.current_file
        return assistant.chapter_map.get(filename, {}) if filename else {}

    @tool
    def get_chapter_content(section_name: str):
        """Get section content. Input: section_name (e.g., unit_1)"""
        filename = assistant.current_file

        if not filename:
            return "No document selected."

        indices = assistant.chapter_map.get(filename, {}).get(section_name, [])

        if not indices:
            return "Section not found."

        chunks = [assistant.chunks[filename][i].page_content for i in indices]

        return "\n".join(chunks)

    @tool
    def search_within_section(data: str):
        """Search inside section. Format: section|query"""
        filename = assistant.current_file

        if not filename:
            return "No document selected."

        parts = data.split("|")

        if len(parts) != 2:
            return "Expected format: section|query"

        section, query = parts

        indices = assistant.chapter_map.get(filename, {}).get(section, [])

        if not indices:
            return "Section not found."

        text = "\n".join(
            assistant.chunks[filename][i].page_content
            for i in indices
        )

        return "\n".join(
            line for line in text.splitlines()
            if query.lower() in line.lower()
        )

    @tool
    def summarize_document_tool():
        """Generate full document summary."""
        return "".join(list(assistant.summarize_document()))

    @tool
    def summarize_section_tool(section_name: str):
        """Summarize a section (e.g., unit_1)."""
        return "".join(list(assistant.summarize_section(section_name)))

    @tool
    def answer_question_tool(question: str):
        """Answer using current document context."""
        filename = assistant.current_file

        if not filename:
            return "No document selected."

        context = assistant.retrieve(question)

        result = assistant.agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }]
        })

        return result["messages"][-1].content

    return [
        retrieve_document_context,
        get_section_map,
        get_chapter_content,
        search_within_section,
        summarize_document_tool,
        summarize_section_tool,
        answer_question_tool
    ]