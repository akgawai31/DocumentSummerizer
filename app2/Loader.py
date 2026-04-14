from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader, CSVLoader

def load_document(file_path, file_type):
    if file_type == "pdf":
        loader = PyMuPDFLoader(
            file_path,
            extract_images=False
        )
    elif file_type == "txt":
        loader = TextLoader(file_path)
    elif file_type == "docx":
        loader = Docx2txtLoader(file_path)
    elif file_type == "csv":
        loader = CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    docs = loader.load()

    # 🔥 ADD PAGE METADATA ENRICHMENT
    for i, d in enumerate(docs):
        d.metadata["page"] = i

    return docs