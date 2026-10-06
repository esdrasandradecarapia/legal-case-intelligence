from ai_research_agent.domain.entities import Chunk, Document


class Chunker:
    def __init__(self, chunk_size: int, overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if overlap < 0:
            raise ValueError("overlap must be greater than or equal to zero")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Document) -> list[Chunk]:
        if len(document.text) == 0:
            return []

        chunks = []
        step = self.chunk_size - self.overlap
        start = 0
        index = 0

        while True:
            piece = document.text[start : start + self.chunk_size]

            chunks.append(
                Chunk(
                    document_id=document.id,
                    index=index,
                    text=piece,
                )
            )

            if start + self.chunk_size >= len(document.text):
                break

            start += step
            index += 1

        return chunks
