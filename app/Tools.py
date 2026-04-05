from langchain.tools import tool

def create_document_tools(agent_instance):
    # -------------------------
    # 🔍 SEARCH TOOL (specific file)
    # -------------------------
    @tool
    def search_document_in_file(query: str, filename: str) -> str:
        """
        Search within a specific document.
        """
        if not hasattr(agent_instance, "retrieve"):
            return "Error: The agent does not have a 'retrieve' method."
        if not hasattr(agent_instance, "vectorstores"):
            return "Error: No processed documents available."

        if filename not in agent_instance.vectorstores:
            return f"Document '{filename}' not processed yet."

        try:
            result = agent_instance.retrieve(query, filename)
            if not result.strip():
                return f"No relevant content found in '{filename}'."
            return result
        except Exception as e:
            return f"Error retrieving from '{filename}': {str(e)}"
        
    @tool
    def summarize_chapter_tool(filename: str, chapter_name: str) -> str:
        """Summarize a specific chapter from a stored document."""
        return self.summarize_chapter(filename, chapter_name)

    # -------------------------
    # 🔍 SEARCH TOOL (all documents)
    # -------------------------
    @tool
    def search_document(query: str) -> str:
        """
        Search relevant content across all processed documents.
        """
        if not hasattr(agent_instance, "retrieve"):
            return "Error: The agent does not have a 'retrieve' method."
        if not hasattr(agent_instance, "vectorstores") or not agent_instance.vectorstores:
            return "No processed documents available for search."

        results = []
        for filename in agent_instance.vectorstores:
            try:
                res = agent_instance.retrieve(query, filename)
                if res.strip():
                    results.append(f"📄 {filename}:\n{res}")
            except Exception as e_file:
                results.append(f"Error retrieving from '{filename}': {str(e_file)}")

        return "\n\n".join(results) if results else "No relevant content found across documents."

    # -------------------------
    # 📝 SUMMARIZE TOOL
    # -------------------------
    @tool
    def summarize_document(filename: str) -> str:
        """
        Generate a structured summary of a specific document.
        """
        if not hasattr(agent_instance, "summarize_document"):
            return "Error: The agent does not have a 'summarize_document' method."
        if not hasattr(agent_instance, "vectorstores") or filename not in agent_instance.vectorstores:
            return f"Document '{filename}' not processed or unavailable."

        try:
            summary = agent_instance.summarize_document(filename)
            if not summary.strip():
                return f"No content available to summarize in '{filename}'."
            return summary
        except Exception as e:
            return f"Error summarizing '{filename}': {str(e)}"
        
    
    @tool
    def search_csv_in_file(query: str, filename: str) -> str:
        """
        Search within a specific CSV file for matching rows.
        """
        if not hasattr(agent_instance, "vectorstores"):
            return "Error: No CSV data loaded in the agent."
        if filename not in agent_instance.vectorstores:
            return f"CSV file '{filename}' not loaded."

        try:
            df = agent_instance.vectorstores[filename]
            mask = df.apply(lambda col: col.astype(str).str.contains(query, case=False, na=False)
                            if col.dtype == "object" else False)
            matching_rows = df[mask.any(axis=1)]
            if matching_rows.empty:
                return f"No matching rows found in '{filename}'."
            return matching_rows.to_csv(index=False)
        except Exception as e:
            return f"Error searching CSV '{filename}': {str(e)}"


    @tool
    def summarize_csv(filename: str) -> str:
        """
        Generate a summary of a CSV file, including basic stats and structure.
        """
        if not hasattr(agent_instance, "vectorstores"):
            return "Error: No CSV data loaded in the agent."
        if filename not in agent_instance.vectorstores:
            return f"CSV file '{filename}' not loaded."

        try:
            df = agent_instance.vectorstores[filename]
            summary = f"CSV '{filename}' Summary:\n"
            summary += f"- Rows: {df.shape[0]}\n- Columns: {df.shape[1]}\n"
            summary += f"- Column names: {', '.join(df.columns)}\n\n"
            summary += "Column Types & Basic Stats:\n"
            summary += str(df.describe(include='all').transpose())
            return summary
        except Exception as e:
            return f"Error summarizing CSV '{filename}': {str(e)}"


    @tool
    def analyze_csv(filename: str, column: str, operation: str = "mean") -> str:
        """
        Perform a simple analysis (sum, mean, min, max) on a numeric column.
        """
        if not hasattr(agent_instance, "vectorstores"):
            return "Error: No CSV data loaded in the agent."
        if filename not in agent_instance.vectorstores:
            return f"CSV file '{filename}' not loaded."

        try:
            df = agent_instance.vectorstores[filename]
            if column not in df.columns:
                return f"Column '{column}' not found in '{filename}'."
            if not pd.api.types.is_numeric_dtype(df[column]):
                return f"Column '{column}' is not numeric and cannot be analyzed."

            ops = {"mean": df[column].mean(),
                "sum": df[column].sum(),
                "min": df[column].min(),
                "max": df[column].max()}
            result = ops.get(operation.lower())
            if result is None:
                return f"Operation '{operation}' not supported. Choose from sum, mean, min, max."
            return f"{operation.capitalize()} of '{column}' in '{filename}' is {result}"
        except Exception as e:
            return f"Error analyzing CSV '{filename}': {str(e)}"

    return [search_document, summarize_chapter_tool, summarize_document, search_document_in_file, search_csv_in_file, analyze_csv, summarize_csv]