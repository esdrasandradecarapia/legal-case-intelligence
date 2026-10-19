import json

from openai import OpenAI

from ai_research_agent.domain.ports import ToolDecision

OLLAMA_BASE_URL = "http://localhost:11434/v1"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"


class OllamaEmbedder:
    def __init__(self) -> None:
        self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding


class OllamaLLM:
    def __init__(self) -> None:
        self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def decide_tool(self, question: str, tools: list[dict]) -> ToolDecision:
        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": question}],
            tools=tools,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return ToolDecision(tool_name=None, arguments={})

        call = message.tool_calls[0]
        arguments = json.loads(call.function.arguments)

        return ToolDecision(tool_name=call.function.name, arguments=arguments)

