from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    name: str
    text: str


@dataclass(frozen=True)
class Chunk:
    document_id: str
    index: int
    text: str

@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Chunk]