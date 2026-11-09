import re
from datetime import datetime
from pathlib import Path

import pytest

from vet_agent.agent.service import AgentService
from vet_agent.llm_local import LocalLLM
from vet_agent.rag.embeddings import HashingEmbedder
from vet_agent.rag.service import FaqService
from vet_agent.rag.store import InMemoryVectorStore
from vet_agent.scheduling.models import AppointmentStatus
from vet_agent.scheduling.repository import InMemoryAppointmentRepository
from vet_agent.scheduling.service import SchedulingService

NOW = datetime(2026, 10, 5, 9, 0)
FAQ_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "faq"


@pytest.fixture
def agent() -> tuple[AgentService, SchedulingService]:
    llm = LocalLLM()
    scheduling = SchedulingService(InMemoryAppointmentRepository(), now=lambda: NOW)
    faq = FaqService(HashingEmbedder(), InMemoryVectorStore(), llm)
    faq.ingest(FAQ_DIRECTORY)
    return AgentService(llm, faq, scheduling, now=lambda: NOW), scheduling


def book_flow(service: AgentService, session: str) -> str:
    service.chat("Quero agendar um banho", session)
    service.chat("Meu nome é Ana", session)
    service.chat("O pet é o cachorro Rex", session)
    service.chat("Dia 07/10 às 10h", session)
    final = service.chat("sim", session)
    return final.reply


def test_faq_question_is_answered_from_the_price_section(agent) -> None:
    service, _ = agent
    result = service.chat("Quanto custa o banho?")
    assert result.intent == "faq"
    assert result.sources[0].source == "servicos.md"
    assert "R$" in result.reply


def test_medical_question_is_refused(agent) -> None:
    service, _ = agent
    assert service.chat("Qual remédio posso dar para meu gato?").intent == "medical_advice"


def test_booking_flow_creates_an_appointment(agent) -> None:
    service, scheduling = agent
    reply = book_flow(service, "s1")
    assert "Agendamento confirmado" in reply
    appointment_id = re.search(r"Código: (\S+)\.", reply).group(1)
    appointment = scheduling.get(appointment_id)
    assert appointment.pet_name == "Rex"
    assert appointment.tutor_name == "Ana"
    assert appointment.start == datetime(2026, 10, 7, 10, 0)


def test_booking_asks_for_missing_details_before_calling_tools(agent) -> None:
    service, _ = agent
    first = service.chat("Quero agendar uma consulta", "s2")
    assert first.tool_calls == []
    assert "nome do tutor" in first.reply


def test_reschedule_and_cancel_flows(agent) -> None:
    service, scheduling = agent
    reply = book_flow(service, "s3")
    appointment_id = re.search(r"Código: (\S+)\.", reply).group(1)

    service.chat(f"Quero remarcar o agendamento {appointment_id}", "s3")
    service.chat("Para dia 08/10 às 14h", "s3")
    moved = service.chat("sim", "s3")
    assert "Agendamento remarcado" in moved.reply
    assert scheduling.get(appointment_id).start == datetime(2026, 10, 8, 14, 0)

    service.chat(f"Quero cancelar o agendamento {appointment_id}", "s3")
    cancelled = service.chat("sim", "s3")
    assert "Agendamento cancelado" in cancelled.reply
    assert scheduling.get(appointment_id).status == AppointmentStatus.CANCELLED


def test_slot_conflict_is_explained(agent) -> None:
    service, _ = agent
    book_flow(service, "s4")
    service.chat("Quero agendar uma consulta", "s5")
    service.chat("Sou o Bruno", "s5")
    service.chat("Meu gato Mia", "s5")
    service.chat("Dia 07/10 às 10h", "s5")
    result = service.chat("sim", "s5")
    assert "Não consegui concluir" in result.reply


def test_list_slots_for_a_day(agent) -> None:
    service, _ = agent
    result = service.chat("Quais horários livres amanhã?")
    assert result.tool_calls[0].name == "list_available_slots"
    assert "Horários livres" in result.reply
