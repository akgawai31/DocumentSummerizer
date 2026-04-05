import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from app.Tools import create_document_tools

load_dotenv()


class GroqClient:
    def __init__(self, model: str = "openai/gpt-oss-120b", temperature: float = 0.3):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set in environment variables")
        self.model = model
        self.client = ChatGroq(
            model=self.model,
            groq_api_key=self.api_key
        )

    # def generate(self, prompt: str) -> str:
    #     response = self.client.invoke(prompt)
    #     return getattr(response, "content", str(response))

    def build_agent(self):
        tools = create_document_tools(self.client)
        agent = create_agent(self.client, tools=tools)
        return agent, tools