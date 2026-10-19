import json

from openai import OpenAI

from ai_research_agent.domain.ports import ToolDecision

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


class OpenAIEmbedder:
    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding


class OpenAILLM:
    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(api_key=api_key)

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