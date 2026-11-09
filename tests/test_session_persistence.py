import re
from datetime import datetime
from pathlib import Path

from vet_agent.agent.checkpointing import postgres_checkpointer
from vet_agent.agent.service import AgentService
from vet_agent.llm_local import LocalLLM
from vet_agent.rag.embeddings import HashingEmbedder
from vet_agent.rag.service import FaqService
from vet_agent.rag.store import InMemoryVectorStore
from vet_agent.scheduling.pgrepository import PostgresAppointmentRepository
from vet_agent.scheduling.repository import InMemoryAppointmentRepository
from vet_agent.scheduling.service import SchedulingService

NOW = datetime(2026, 10, 5, 9, 0)
FAQ_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "faq"


def start_api_process(database_url: str | None) -> AgentService:
    llm = LocalLLM()
    repository = (
        PostgresAppointmentRepository(database_url)
        if database_url
        else InMemoryAppointmentRepository()
    )
    faq = FaqService(HashingEmbedder(), InMemoryVectorStore(), llm)
    faq.ingest(FAQ_DIRECTORY)
    return AgentService(
        llm,
        faq,
        SchedulingService(repository, now=lambda: NOW),
        now=lambda: NOW,
        checkpointer=postgres_checkpointer(database_url) if database_url else None,
    )


def converse_until_confirmation(service: AgentService, session: str) -> None:
    service.chat("Quero agendar uma consulta", session)
    service.chat("Meu nome é Ana", session)
    service.chat("Meu cachorro Rex", session)
    reply = service.chat("Dia 07/10 às 10h", session)
    assert reply.reply.startswith("Confirma o agendamento")


def test_conversation_survives_an_api_restart(postgres_url: str) -> None:
    before_restart = start_api_process(postgres_url)
    converse_until_confirmation(before_restart, "tutor-1")

    after_restart = start_api_process(postgres_url)
    final = after_restart.chat("sim", "tutor-1")

    assert "Agendamento confirmado" in final.reply
    appointment_id = re.search(r"Código: (\S+)\.", final.reply).group(1)
    stored = SchedulingService(PostgresAppointmentRepository(postgres_url)).get(appointment_id)
    assert stored.pet_name == "Rex"
    assert stored.tutor_name == "Ana"


def test_sessions_are_isolated_by_session_id(postgres_url: str) -> None:
    service = start_api_process(postgres_url)
    converse_until_confirmation(service, "tutor-1")

    other = service.chat("sim", "tutor-2")

    assert "Agendamento confirmado" not in other.reply


def test_memory_sessions_do_not_survive_a_restart() -> None:
    before_restart = start_api_process(None)
    converse_until_confirmation(before_restart, "tutor-1")

    after_restart = start_api_process(None)
    final = after_restart.chat("sim", "tutor-1")

    assert "Agendamento confirmado" not in final.reply
