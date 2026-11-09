import json
from collections import Counter
from pathlib import Path

import pytest

from vet_agent import evaluation_routing
from vet_agent.agent.graph import INTENTS
from vet_agent.evaluation_retrieval import build_index, evaluate_retrieval
from vet_agent.evaluation_retrieval import load_cases as load_faq_cases
from vet_agent.evaluation_routing import evaluate_routing
from vet_agent.evaluation_routing import load_cases as load_routing_cases
from vet_agent.rag.chunking import load_directory
from vet_agent.rag.embeddings import HashingEmbedder
from vet_agent.rag.store import InMemoryVectorStore

ROOT = Path(__file__).resolve().parent.parent
FAQ_CASES = ROOT / "data" / "eval" / "faq.json"
ROUTING_CASES = ROOT / "data" / "eval" / "routing.json"
FAQ_DIRECTORY = ROOT / "data" / "faq"

MIN_HIT_AT_3 = 0.85


class ScriptedClassifier:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def complete(self, system: str, user: str) -> str:
        message = user.removeprefix("Tutor: ")
        return json.dumps({"intent": self.mapping.get(message, "smalltalk")})


def test_routing_dataset_has_thirty_valid_labeled_messages():
    cases = load_routing_cases(ROUTING_CASES)

    assert len(cases) == 30
    assert {case["intent"] for case in cases} == INTENTS
    assert len({case["message"] for case in cases}) == 30
    assert min(Counter(case["intent"] for case in cases).values()) >= 6


def test_routing_runner_scores_a_perfect_classifier():
    cases = load_routing_cases(ROUTING_CASES)
    llm = ScriptedClassifier({case["message"]: case["intent"] for case in cases})

    report = evaluate_routing(cases, llm)

    assert report["accuracy"] == 1.0
    assert report["errors"] == []
    assert all(stats["recall"] == 1.0 for stats in report["per_intent"].values())


def test_routing_runner_reports_the_errors_of_a_weak_classifier():
    cases = load_routing_cases(ROUTING_CASES)

    report = evaluate_routing(cases, ScriptedClassifier({}))

    assert report["accuracy"] == pytest.approx(6 / 30)
    assert report["per_intent"]["faq"]["recall"] == 0.0
    assert report["per_intent"]["smalltalk"]["recall"] == 1.0
    assert {"message", "expected", "predicted"} == set(report["errors"][0])


def test_routing_runner_needs_credentials(monkeypatch, capsys):
    from vet_agent.config import Settings

    monkeypatch.setattr(
        evaluation_routing,
        "get_settings",
        lambda: Settings(azure_openai_endpoint="", azure_openai_api_key="", _env_file=None),
    )

    assert evaluation_routing.main() == 2
    assert "AZURE_OPENAI" in capsys.readouterr().err


def test_faq_dataset_points_to_existing_documents_and_sections():
    cases = load_faq_cases(FAQ_CASES)
    titles = {(chunk.source, chunk.title) for chunk in load_directory(FAQ_DIRECTORY)}

    assert len(cases) == 30
    for case in cases:
        assert any(
            source == case["source"] and case["section"] in title for source, title in titles
        ), case


def test_faq_retrieval_meets_the_regression_threshold_offline():
    embedder = HashingEmbedder()
    store = InMemoryVectorStore()
    build_index(FAQ_DIRECTORY, embedder, store)

    report = evaluate_retrieval(load_faq_cases(FAQ_CASES), embedder, store)

    assert report["hit@3"] >= MIN_HIT_AT_3
    assert report["hit@1"] <= report["hit@3"]


def test_faq_runner_counts_misses():
    class EmptyStore(InMemoryVectorStore):
        pass

    embedder = HashingEmbedder()
    store = EmptyStore()
    build_index(FAQ_DIRECTORY, embedder, store)
    cases = [
        {"question": "Qual o preço da astronomia?", "source": "servicos.md", "section": "Exames"}
    ]

    report = evaluate_retrieval(cases, embedder, store)

    assert report["total"] == 1
    assert report["hit@3"] == 0.0
    assert report["misses"][0]["expected"] == "servicos.md / Exames"
