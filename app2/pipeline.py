import re
import tempfile

from app2.GroqAgent import GroqClient
from app2.Loader import load_document
from app2.MetaData import Document
from app2.Processing import split_documents, get_embeddings, create_vectorstore
from app2.Tools import create_document_tools


class SmartDocAssistant:

    def __init__(self):
        self.documents = {}
        self.vectorstores = {}
        self.chunks = {}
        self.chapter_map = {}
        self.summary_cache = {}
        self.current_file = None
        self.section_order = []

        self.model = GroqClient()
        self.llm = self.model.get_llm()

        self.tools = create_document_tools(self)
        self.agent = self.model.build_agent(self.tools)

        print("Assistant initialized")

    
    #Load Document
    def load_document_file(self, uploaded_file):
        file_type = uploaded_file.name.split(".")[-1]

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        raw_docs = load_document(tmp_path, file_type)

        docs = []
        for i, doc in enumerate(raw_docs):
            metadata = getattr(doc, "metadata", {}).copy()
            metadata.update({
                "filename": uploaded_file.name,
                "chunk_index": i
            })

            docs.append(Document(
                page_content=doc.page_content,
                metadata=metadata
            ))

        self.documents[uploaded_file.name] = docs
        return docs

    
    def normalize(self, t, n):
        roman = {
            "i": "1", "ii": "2", "iii": "3",
            "iv": "4", "v": "5", "vi": "6",
            "vii": "7", "viii": "8", "ix": "9", "x": "10"
        }

        word = {
            "one": "1", "two": "2", "three": "3",
            "four": "4", "five": "5", "six": "6",
            "seven": "7", "eight": "8", "nine": "9", "ten": "10"
        }

        if not n:
            return t.lower()

        n = n.lower().strip()
        n = roman.get(n, n)
        n = word.get(n, n)

        return f"{t.lower()}_{n}"

    
    def sort_sections(self, order):
        def extract_num(s):
            nums = re.findall(r"\d+", s)
            return int(nums[0]) if nums else 999
        return sorted(order, key=extract_num)

    
    #Build Table of content from chunks
    def build_toc_from_chunks(self, filename):
        chunks = self.chunks.get(filename, [])

        toc = {"introduction": []}
        order = []

        pattern = re.compile(
            r"\b(unit|chapter|section|module)\b\s*[-–—]?\s*"
            r"([0-9]+|[ivx]+|one|two|three|four|five|six|seven|eight|nine|ten)",
            re.IGNORECASE
        )

        current = "introduction"

        for i, chunk in enumerate(chunks):
            text = chunk.page_content.lower()

            # merge next chunk
            if i < len(chunks) - 1:
                text += " " + chunks[i + 1].page_content.lower()

            match = pattern.search(text)

            if match:
                current = self.normalize(match.group(1), match.group(2))

                if current not in toc:
                    toc[current] = []
                    order.append(current)

            toc[current].append(i)

        order = self.sort_sections(order)

        return toc, order

    #Process document
    def process_document(self, filename):
        self.current_file = filename

        chunks = split_documents(self.documents[filename])
        self.chunks[filename] = chunks

        toc, order = self.build_toc_from_chunks(filename)

        self.chapter_map[filename] = toc
        self.section_order = order

        embeddings = get_embeddings()
        self.vectorstores[filename] = create_vectorstore(chunks, embeddings)
        return "Document processed"

    #Retrival
    def retrieve(self, query):
        filename = self.current_file

        if not filename:
            return ""

        vs = self.vectorstores.get(filename)
        if not vs:
            return ""

        docs = vs.similarity_search(query, k=3)
        return "\n\n".join([d.page_content for d in docs])

    
    #get count
    def get_structure_count(self, key):
        filename = self.current_file

        items = [
            k for k in self.chapter_map.get(filename, {})
            if k.startswith(key)
        ]

        return iter([f"The document contains {len(items)} {key}s."])

    #section summary
    def summarize_section(self, section_name):
        filename = self.current_file

        indices = self.chapter_map.get(filename, {}).get(section_name, [])

        if not indices:
            return iter([f"Section '{section_name}' not found."])

        text = "\n".join(
            self.chunks[filename][i].page_content for i in indices
        )

        prompt = f"""
            Summarize ONLY this section: {section_name}

            {text}
        """

        return iter([self.llm.invoke(prompt).content])

    #Document Summary
    def summarize_document(self):
        filename = self.current_file

        if filename in self.summary_cache:
            return iter([self.summary_cache[filename]])

        chunks = self.chunks.get(filename, [])
        grouped = [chunks[i:i+3] for i in range(0, len(chunks), 3)]

        partials = []

        for group in grouped:
            text = "\n".join(c.page_content for c in group)
            res = self.llm.invoke(f"Summarize:\n{text}")
            partials.append(res.content)

        final = self.llm.invoke("\n".join(partials)).content

        self.summary_cache[filename] = final

        return iter([final])

    
    #QA
    def stream_qa(self, question):
        context = self.retrieve(question)

        result = self.agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }]
        })

        return result["messages"][-1].content

    
    #main ask
    def ask(self, question):
        if "how many units" in question.lower():
            return self.get_structure_count("unit")

        if "summary" in question.lower():
            return self.summarize_document()

        return iter([self.stream_qa(question)])

    def ask_sync(self, question):
        return "".join(list(self.ask(question)))