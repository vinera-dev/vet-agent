import os
from pathlib import Path

import pytest

from vet_agent.rag.embeddings import HashingEmbedder
from vet_agent.rag.pgstore import PgVectorStore
from vet_agent.rag.service import FaqService

DATABASE_URL = os.environ.get("PGVECTOR_TEST_URL")
FAQ_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "faq"

pytestmark = pytest.mark.skipif(DATABASE_URL is None, reason="PGVECTOR_TEST_URL não definida")


class FakeLLM:
    def complete(self, system: str, user: str) -> str:
        return "resposta gerada"


def test_pgvector_store_ranks_like_the_in_memory_store() -> None:
    embedder = HashingEmbedder()
    store = PgVectorStore(DATABASE_URL, embedder.dimensions)
    service = FaqService(embedder, store, FakeLLM())

    assert service.ingest(FAQ_DIRECTORY) > 0
    answer = service.ask("Meu cachorro teve convulsão, o que faço?")

    assert answer.grounded is True
    assert answer.sources[0].source == "urgencias.md"


def test_pgvector_ingest_is_idempotent() -> None:
    embedder = HashingEmbedder()
    store = PgVectorStore(DATABASE_URL, embedder.dimensions)
    service = FaqService(embedder, store, FakeLLM())

    first = service.ingest(FAQ_DIRECTORY)
    second = service.ingest(FAQ_DIRECTORY)

    assert first == second
    assert len(store.search(embedder.embed(["vacina"])[0], 100)) == first
