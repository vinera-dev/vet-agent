import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from vet_agent.rag.models import Chunk, SearchResult

TABLE = "faq_chunks"


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        self._database_url = database_url
        self._dimensions = dimensions
        self._prepared = False

    def _connect(self) -> psycopg.Connection:
        connection = psycopg.connect(self._database_url, autocommit=True)
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(connection)
        return connection

    def _ensure_table(self, connection: psycopg.Connection) -> None:
        if self._prepared:
            return
        connection.execute(
            f"CREATE TABLE IF NOT EXISTS {TABLE} ("
            "id TEXT PRIMARY KEY, source TEXT NOT NULL, title TEXT NOT NULL, "
            f"content TEXT NOT NULL, embedding vector({self._dimensions}) NOT NULL)"
        )
        self._prepared = True

    def reset(self) -> None:
        with self._connect() as connection:
            self._ensure_table(connection)
            connection.execute(f"TRUNCATE {TABLE}")

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        with self._connect() as connection:
            self._ensure_table(connection)
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                connection.execute(
                    f"INSERT INTO {TABLE} (id, source, title, content, embedding) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (id) DO UPDATE SET source = EXCLUDED.source, "
                    "title = EXCLUDED.title, content = EXCLUDED.content, "
                    "embedding = EXCLUDED.embedding",
                    (chunk.id, chunk.source, chunk.title, chunk.content, Vector(embedding)),
                )

    def search(self, embedding: list[float], limit: int) -> list[SearchResult]:
        with self._connect() as connection:
            self._ensure_table(connection)
            rows = connection.execute(
                f"SELECT id, source, title, content, 1 - (embedding <=> %s) AS score "
                f"FROM {TABLE} ORDER BY embedding <=> %s LIMIT %s",
                (Vector(embedding), Vector(embedding), limit),
            ).fetchall()
        return [
            SearchResult(
                chunk=Chunk(id=row[0], source=row[1], title=row[2], content=row[3]),
                score=float(row[4]),
            )
            for row in rows
        ]
