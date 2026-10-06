from typing import Protocol

from ai_research_agent.domain.entities import Chunk


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    def add(
        self,
        chunk: Chunk,
        embedding: list[float],
    ) -> None: ...

    def search(
        self,
        embedding: list[float],
        limit: int,
    ) -> list[Chunk]: ...


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...
