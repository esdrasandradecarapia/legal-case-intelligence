from dataclasses import dataclass
from typing import Protocol

from ai_research_agent.domain.entities import Chunk, RetrievedChunk


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


class VectorStore(Protocol):
    def add(self, chunk: Chunk, embedding: list[float]) -> None:
        ...

    def search(self, embedding: list[float], limit: int) -> list[RetrievedChunk]:
        ...


class LLM(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class OrderStatus:
    order_id: str
    status: str
    eta_days: int


class OrderRepository(Protocol):
    def get_order_status(self, order_id: str) -> OrderStatus:
        ...


class OrderNotFoundError(Exception):
    def __init__(self, order_id: str) -> None:
        self.order_id = order_id
        super().__init__(f"Order not found: {order_id}")

@dataclass(frozen=True)
class ToolDecision:
    tool_name: str | None
    arguments: dict


class ToolCallingLLM(Protocol):
    def decide_tool(self, question: str, tools: list[dict]) -> ToolDecision:
        ...