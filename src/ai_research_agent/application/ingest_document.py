from ai_research_agent.domain.chunking import Chunker
from ai_research_agent.domain.entities import Document
from ai_research_agent.domain.ports import Embedder, VectorStore


class IngestDocument:
    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def execute(self, document: Document) -> None:
        chunks = self.chunker.split(document)

        for chunk in chunks:
            embedding = self.embedder.embed(chunk.text)
            self.vector_store.add(chunk, embedding)
