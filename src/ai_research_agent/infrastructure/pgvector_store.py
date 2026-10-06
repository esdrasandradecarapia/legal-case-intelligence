import pg8000.dbapi

from ai_research_agent.domain.entities import Chunk, RetrievedChunk


class PgVectorStore:
    def __init__(self, host: str, port: int, user: str, password: str, database: str) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database

    def _connect(self):
        return pg8000.dbapi.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
        )

    def add(self, chunk: Chunk, embedding: list[float]) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO chunks (document_id, chunk_index, text, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (chunk.document_id, chunk.index, chunk.text, str(embedding)),
            )
            conn.commit()

    def search(self, embedding: list[float], limit: int) -> list[RetrievedChunk]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT document_id, chunk_index, text, 1 - (embedding <=> %s) AS score
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (str(embedding), str(embedding), limit),
            )
            rows = cursor.fetchall()

        return [
            RetrievedChunk(
                chunk=Chunk(document_id=row[0], index=row[1], text=row[2]),
                score=row[3],
            )
            for row in rows
        ]