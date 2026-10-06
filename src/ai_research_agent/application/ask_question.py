from ai_research_agent.domain.entities import Answer
from ai_research_agent.domain.ports import LLM, Embedder, VectorStore

RETRIEVAL_LIMIT = 4


class AskQuestion:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        llm: LLM,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    def execute(self, question: str) -> Answer:
        embedding = self.embedder.embed(question)

        retrieved_chunks = self.vector_store.search(embedding, limit=RETRIEVAL_LIMIT)

        context = "\n\n".join(item.chunk.text for item in retrieved_chunks)
        prompt = (
            "Responda à pergunta usando apenas o contexto abaixo.\n\n"
            f"Contexto:\n{context}\n\n"
            f"Pergunta: {question}"
        )

        response_text = self.llm.generate(prompt)

        sources = [item.chunk for item in retrieved_chunks]

        return Answer(text=response_text, sources=sources)
