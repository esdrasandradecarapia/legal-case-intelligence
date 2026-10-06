from ai_research_agent.application.ask_question import AskQuestion
from ai_research_agent.domain.entities import Chunk, RetrievedChunk


class FakeEmbedder:
    def __init__(self):
        self.received_texts = []

    def embed(self, text: str) -> list[float]:
        self.received_texts.append(text)
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self, results: list[RetrievedChunk]):
        self.results = results
        self.received_embeddings = []
        self.received_limits = []

    def search(self, embedding: list[float], limit: int) -> list[RetrievedChunk]:
        self.received_embeddings.append(embedding)
        self.received_limits.append(limit)
        return self.results


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.received_prompts = []

    def generate(self, prompt: str) -> str:
        self.received_prompts.append(prompt)
        return self.response


def test_ask_question_returns_answer_with_sources():
    chunk_1 = Chunk(
        document_id="document-1",
        index=0,
        text="A rescisão deve ser notificada com 30 dias.",
    )

    chunk_2 = Chunk(
        document_id="document-1",
        index=1,
        text="Multa de 20% sobre o valor restante do contrato.",
    )

    retrieved_chunks = [
        RetrievedChunk(chunk=chunk_1, score=0.91),
        RetrievedChunk(chunk=chunk_2, score=0.84),
    ]

    embedder = FakeEmbedder()
    vector_store = FakeVectorStore(results=retrieved_chunks)
    llm = FakeLLM(response="Resposta gerada com base no contexto.")

    use_case = AskQuestion(
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
    )

    answer = use_case.execute("O que diz o contrato sobre rescisão?")

    assert answer.text == "Resposta gerada com base no contexto."
    assert answer.sources == [chunk_1, chunk_2]
