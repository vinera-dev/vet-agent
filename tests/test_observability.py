import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from vet_agent import llm
from vet_agent.agent.service import AgentService
from vet_agent.config import Settings
from vet_agent.llm_local import LocalLLM
from vet_agent.observability import TracedLLM, Tracer
from vet_agent.rag.embeddings import HashingEmbedder
from vet_agent.rag.service import FaqService
from vet_agent.rag.store import InMemoryVectorStore
from vet_agent.scheduling.repository import InMemoryAppointmentRepository
from vet_agent.scheduling.service import SchedulingService

NOW = datetime(2026, 10, 5, 9, 0)
FAQ_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "faq"
NODE_NAMES = {
    "guard_input",
    "classify",
    "faq_answer",
    "medical_refusal",
    "smalltalk",
    "plan_scheduling",
    "run_tool",
    "finalize",
}


class UsageReportingLLM(LocalLLM):
    def complete(self, system: str, user: str) -> str:
        reply = super().complete(system, user)
        llm._usage.set(llm.TokenUsage(prompt_tokens=120, completion_tokens=30))
        return reply


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter: InMemorySpanExporter) -> Tracer:
    client = Langfuse(
        public_key=f"pk-lf-{uuid4().hex}",
        secret_key="sk-lf-test",
        host="http://127.0.0.1:1",
        span_exporter=exporter,
    )
    return Tracer(client)


def build_agent(tracer: Tracer) -> AgentService:
    base = UsageReportingLLM()
    llm_client = TracedLLM(base, tracer, "test-deployment")
    scheduling = SchedulingService(InMemoryAppointmentRepository(), now=lambda: NOW)
    faq = FaqService(HashingEmbedder(), InMemoryVectorStore(), llm_client)
    faq.ingest(FAQ_DIRECTORY)
    return AgentService(llm_client, faq, scheduling, now=lambda: NOW, tracer=tracer)


def spans_after(tracer: Tracer, exporter: InMemorySpanExporter) -> list:
    tracer.flush()
    return list(exporter.get_finished_spans())


def test_a_turn_produces_one_trace_with_a_span_per_node(
    tracer: Tracer, exporter: InMemorySpanExporter
) -> None:
    agent = build_agent(tracer)
    agent.chat("Quanto custa o banho?", "sessao-1")
    spans = spans_after(tracer, exporter)

    by_name = {span.name: span for span in spans}
    assert {"agent.chat", "classify", "faq_answer", "finalize", "guard_input"} <= set(by_name)
    assert len({span.context.trace_id for span in spans}) == 1
    root = by_name["agent.chat"]
    assert root.parent is None
    for name in ("guard_input", "classify", "faq_answer", "finalize"):
        assert by_name[name].end_time >= by_name[name].start_time
        assert by_name[name].context.trace_id == root.context.trace_id
    assert {span.name for span in spans} - {"agent.chat", "llm.complete"} <= NODE_NAMES


def test_llm_calls_are_generations_under_their_node_with_token_usage(
    tracer: Tracer, exporter: InMemorySpanExporter
) -> None:
    agent = build_agent(tracer)
    agent.chat("Quanto custa o banho?", "sessao-1")
    spans = spans_after(tracer, exporter)

    by_id = {span.context.span_id: span for span in spans}
    generations = [span for span in spans if span.name == "llm.complete"]
    assert len(generations) == 2
    parents = {by_id[span.parent.span_id].name for span in generations}
    assert parents == {"classify", "faq_answer"}
    for generation in generations:
        assert generation.attributes["langfuse.observation.type"] == "generation"
        assert generation.attributes["langfuse.observation.model.name"] == "test-deployment"
        usage = json.loads(generation.attributes["langfuse.observation.usage_details"])
        assert usage == {"input": 120, "output": 30}


def test_session_id_is_attached_and_raw_personal_data_is_not_recorded(
    tracer: Tracer, exporter: InMemorySpanExporter
) -> None:
    agent = build_agent(tracer)
    agent.chat("Meu CPF é 123.456.789-09, quanto custa o banho?", "sessao-9")
    spans = spans_after(tracer, exporter)

    assert any(span.attributes.get("session.id") == "sessao-9" for span in spans)
    serialized = json.dumps([dict(span.attributes) for span in spans], default=str)
    assert "123.456.789-09" not in serialized
    assert "[CPF]" in serialized


def test_tracing_is_a_noop_without_a_client() -> None:
    agent = build_agent(Tracer())
    result = agent.chat("Quanto custa o banho?", "sessao-1")
    assert result.sources[0].source == "servicos.md"


def test_azure_client_records_the_token_usage_of_each_call() -> None:
    def create(**_: object) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="oi"))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )

    client = llm.AzureOpenAIClient(Settings(azure_openai_deployment="deployment"))
    completions = SimpleNamespace(create=create)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    llm.reset_usage()
    assert client.complete("sistema", "usuário") == "oi"
    assert llm.last_usage() == llm.TokenUsage(prompt_tokens=11, completion_tokens=7)
