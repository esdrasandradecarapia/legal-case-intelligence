from ai_research_agent.domain.entities import Chunk
from ai_research_agent.domain.ports import Embedder, LLM, VectorStore


class FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def add(
        self,
        chunk: Chunk,
        embedding: list[float],
    ) -> None:
        pass

    def search(
        self,
        embedding: list[float],
        limit: int,
    ) -> list[Chunk]:
        return []


class FakeLLM:
    def generate(self, prompt: str) -> str:
        return "fake response"


def test_fake_embedder_matches_port():
    embedder: Embedder = FakeEmbedder()

    assert embedder.embed("hello") == [0.1, 0.2, 0.3]


def test_fake_vector_store_matches_port():
    store: VectorStore = FakeVectorStore()

    chunk = Chunk(
        document_id="1",
        index=0,
        text="hello",
    )

    store.add(chunk, [0.1, 0.2, 0.3])

    assert store.search([0.1, 0.2, 0.3], limit=5) == []


def test_fake_llm_matches_port():
    llm: LLM = FakeLLM()

    assert llm.generate("hello") == "fake response"
