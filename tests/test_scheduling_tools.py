from datetime import datetime

import pytest

from vet_agent.scheduling.repository import InMemoryAppointmentRepository
from vet_agent.scheduling.service import SchedulingService
from vet_agent.scheduling.tools import execute_tool, tool_definitions

NOW = datetime(2026, 10, 5, 9, 0)


@pytest.fixture
def service() -> SchedulingService:
    return SchedulingService(InMemoryAppointmentRepository(), now=lambda: NOW)


def test_tool_definitions_follow_function_calling_contract() -> None:
    definitions = tool_definitions()
    names = {item["function"]["name"] for item in definitions}
    assert names == {
        "list_available_slots",
        "book_appointment",
        "reschedule_appointment",
        "cancel_appointment",
    }
    for item in definitions:
        assert item["type"] == "function"
        assert item["function"]["parameters"]["type"] == "object"
        assert item["function"]["description"]


def test_list_available_slots(service: SchedulingService) -> None:
    result = execute_tool(service, "list_available_slots", {"day": "2026-10-06"})
    assert result["ok"] is True
    assert len(result["result"]) == 20


def test_full_booking_flow(service: SchedulingService) -> None:
    booked = execute_tool(
        service,
        "book_appointment",
        {
            "tutor_name": "Ana",
            "pet_name": "Rex",
            "service": "vacina",
            "start": "2026-10-06T10:00:00",
        },
    )
    assert booked["ok"] is True
    appointment_id = booked["result"]["id"]

    moved = execute_tool(
        service,
        "reschedule_appointment",
        {"appointment_id": appointment_id, "new_start": "2026-10-06T11:00:00"},
    )
    assert moved["result"]["start"] == "2026-10-06T11:00:00"

    cancelled = execute_tool(service, "cancel_appointment", {"appointment_id": appointment_id})
    assert cancelled["result"]["status"] == "cancelled"


def test_business_error_is_returned_as_structured_result(service: SchedulingService) -> None:
    result = execute_tool(
        service,
        "book_appointment",
        {
            "tutor_name": "Ana",
            "pet_name": "Rex",
            "service": "consulta",
            "start": "2026-10-11T10:00:00",
        },
    )
    assert result["ok"] is False
    assert result["error"]["code"] == "outside_business_hours"


def test_invalid_arguments_are_reported(service: SchedulingService) -> None:
    result = execute_tool(service, "book_appointment", {"tutor_name": "Ana"})
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_arguments"


def test_unknown_tool_is_reported(service: SchedulingService) -> None:
    result = execute_tool(service, "delete_everything", {})
    assert result["ok"] is False
    assert result["error"]["code"] == "unknown_tool"
