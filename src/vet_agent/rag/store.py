import math
from typing import Protocol

from vet_agent.rag.models import Chunk, SearchResult


class VectorStore(Protocol):
    def reset(self) -> None: ...

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def search(self, embedding: list[float], limit: int) -> list[SearchResult]: ...


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    def reset(self) -> None:
        self._items.clear()

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self._items[chunk.id] = (chunk, embedding)

    def search(self, embedding: list[float], limit: int) -> list[SearchResult]:
        results = [
            SearchResult(chunk=chunk, score=cosine_similarity(embedding, stored))
            for chunk, stored in self._items.values()
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:limit]
