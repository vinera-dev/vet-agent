from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vet_agent.dependencies import get_llm
from vet_agent.main import app
from vet_agent.rag.chunking import chunk_markdown, load_directory
from vet_agent.rag.embeddings import HashingEmbedder
from vet_agent.rag.routes import get_faq_service
from vet_agent.rag.service import FAQ_SYSTEM_PROMPT, NO_ANSWER_MESSAGE, FaqService
from vet_agent.rag.store import InMemoryVectorStore, cosine_similarity

FAQ_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "faq"


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return "resposta gerada"


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def faq(llm: FakeLLM) -> FaqService:
    service = FaqService(HashingEmbedder(), InMemoryVectorStore(), llm)
    service.ingest(FAQ_DIRECTORY)
    return service


def test_chunking_splits_by_section_and_keeps_titles() -> None:
    text = "# Doc\n\n## Primeira\nconteudo um\n\n## Segunda\nconteudo dois\n"
    chunks = chunk_markdown("doc.md", text)
    assert [chunk.id for chunk in chunks] == ["doc.md#0", "doc.md#1"]
    assert chunks[0].title == "Doc - Primeira"
    assert "conteudo um" in chunks[0].content
    assert "conteudo dois" not in chunks[0].content


def test_chunking_skips_empty_sections() -> None:
    assert chunk_markdown("doc.md", "# Doc\n\n## Vazia\n\n## Cheia\ntexto\n")[0].id == "doc.md#1"


def test_load_directory_reads_every_faq_file() -> None:
    chunks = load_directory(FAQ_DIRECTORY)
    sources = {chunk.source for chunk in chunks}
    assert sources == {
        "horarios.md",
        "servicos.md",
        "vacinas.md",
        "urgencias.md",
        "politicas.md",
    }


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder()
    first, second = embedder.embed(["Banho e tosa", "banho e tosa"])
    assert first == second
    assert cosine_similarity(first, second) == pytest.approx(1.0)


def test_hashing_embedder_ignores_accents_and_case() -> None:
    embedder = HashingEmbedder()
    plain, accented = embedder.embed(["acacias horario", "ACÁCIAS Horário"])
    assert plain == accented


@pytest.mark.parametrize(
    ("question", "expected_source"),
    [
        ("A clínica atende aos domingos?", "horarios.md"),
        ("Quanto custa o banho?", "servicos.md"),
        ("Com quantos dias o filhote toma a primeira vacina V10?", "vacinas.md"),
        ("Meu cachorro teve convulsão, o que faço?", "urgencias.md"),
        ("Posso cancelar a consulta em cima da hora?", "politicas.md"),
    ],
)
def test_retrieval_returns_the_right_document_first(
    faq: FaqService, question: str, expected_source: str
) -> None:
    answer = faq.ask(question)
    assert answer.grounded is True
    assert answer.sources[0].source == expected_source


def test_grounded_answer_sends_context_to_the_llm(faq: FaqService, llm: FakeLLM) -> None:
    answer = faq.ask("Quanto custa o banho?")
    assert answer.answer == "resposta gerada"
    system, user = llm.calls[0]
    assert system == FAQ_SYSTEM_PROMPT
    assert "<contexto>" in user and "<pergunta>" in user
    assert "R$ 60,00" in user


def test_question_without_relevant_context_skips_the_llm(faq: FaqService, llm: FakeLLM) -> None:
    answer = faq.ask("Qual a capital da Mongólia?")
    assert answer.grounded is False
    assert answer.answer == NO_ANSWER_MESSAGE
    assert answer.sources == []
    assert llm.calls == []


def test_ingest_replaces_previous_content(llm: FakeLLM, tmp_path: Path) -> None:
    store = InMemoryVectorStore()
    service = FaqService(HashingEmbedder(), store, llm)
    service.ingest(FAQ_DIRECTORY)
    (tmp_path / "novo.md").write_text("# Novo\n\n## Unico\nvacina de teste\n", encoding="utf-8")
    assert service.ingest(tmp_path) == 1
    assert service.ask("Quanto custa o banho?").grounded is False


@pytest.fixture
def client(faq: FaqService) -> Iterator[TestClient]:
    app.dependency_overrides[get_faq_service] = lambda: faq
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_ask_endpoint(client: TestClient) -> None:
    response = client.post("/faq/ask", json={"question": "Quanto custa o banho?"})
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["sources"][0]["source"] == "servicos.md"


def test_ask_endpoint_rejects_empty_question(client: TestClient) -> None:
    assert client.post("/faq/ask", json={"question": ""}).status_code == 422


def test_ingest_endpoint_reports_chunks(llm: FakeLLM) -> None:
    service = FaqService(HashingEmbedder(), InMemoryVectorStore(), llm)
    app.dependency_overrides[get_faq_service] = lambda: service
    app.dependency_overrides[get_llm] = lambda: llm
    try:
        response = TestClient(app).post("/faq/ingest")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["chunks"] > 0
