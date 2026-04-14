import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()


class GroqClient:

    def __init__(self, model: str = "openai/gpt-oss-120b", temperature: float = 0.3):
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set")

        self.client = ChatGroq(
            model=model,
            groq_api_key=self.api_key,
            temperature=temperature
        )

    def get_llm(self):
        return self.client  # raw LLM without tools

    #Build Agent
    def build_agent(self, tools):

        agent = create_react_agent(
            model=self.client,
            tools=tools
        )

        return agent