from ai_research_agent.application.ingest_document import IngestDocument
from ai_research_agent.domain.chunking import Chunker
from ai_research_agent.domain.entities import Document


class FakeEmbedder:
    def __init__(self):
        self.received_texts = []

    def embed(self, text: str) -> list[float]:
        self.received_texts.append(text)
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self):
        self.received_chunks = []

    def add(self, chunk, embedding):
        self.received_chunks.append((chunk, embedding))


def test_ingest_document_chunks_embeds_and_stores():

    document = Document(
        id="document-1",
        name="case.txt",
        text="abcdefghij",
    )

    chunker = Chunker(chunk_size=4, overlap=1)
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    use_case = IngestDocument(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    use_case.execute(document)

    assert embedder.received_texts == [
        "abcd",
        "defg",
        "ghij",
    ]

    assert len(vector_store.received_chunks) == 3

    assert vector_store.received_chunks[0][0].text == "abcd"
    assert vector_store.received_chunks[0][1] == [0.1, 0.2, 0.3]

    def test_ingest_document_with_empty_text_does_nothing():

        document = Document(
            id="document-1",
            name="case.txt",
            text="",
        )

        chunker = Chunker(chunk_size=4, overlap=1)
        embedder = FakeEmbedder()
        vector_store = FakeVectorStore()

        use_case = IngestDocument(
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
        )

        use_case.execute(document)

        assert embedder.received_texts == []
        assert vector_store.received_chunks == []


class PredictableFakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [float(len(text))]


def test_ingest_document_stores_each_chunk_with_its_own_embedding():
    document = Document(
        id="document-1",
        name="case.txt",
        text="abcdefghij",
    )

    chunker = Chunker(chunk_size=4, overlap=1)
    embedder = PredictableFakeEmbedder()
    vector_store = FakeVectorStore()

    use_case = IngestDocument(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    use_case.execute(document)

    for chunk, embedding in vector_store.received_chunks:
        assert embedding == embedder.embed(chunk.text)


def test_ingest_document_with_empty_text_does_nothing():
    document = Document(
        id="document-1",
        name="case.txt",
        text="",
    )

    chunker = Chunker(chunk_size=4, overlap=1)
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    use_case = IngestDocument(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    use_case.execute(document)

    assert embedder.received_texts == []
    assert vector_store.received_chunks == []
