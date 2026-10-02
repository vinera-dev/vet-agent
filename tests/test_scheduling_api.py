from collections.abc import Iterator
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from vet_agent.main import app
from vet_agent.scheduling.repository import InMemoryAppointmentRepository
from vet_agent.scheduling.routes import get_scheduling_service
from vet_agent.scheduling.service import SchedulingService

NOW = datetime(2026, 10, 5, 9, 0)


@pytest.fixture
def client() -> Iterator[TestClient]:
    service = SchedulingService(InMemoryAppointmentRepository(), now=lambda: NOW)
    app.dependency_overrides[get_scheduling_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


def payload(start: str = "2026-10-06T10:00:00") -> dict[str, str]:
    return {"tutor_name": "Ana", "pet_name": "Rex", "service": "consulta", "start": start}


def test_list_slots(client: TestClient) -> None:
    response = client.get("/appointments/slots", params={"day": "2026-10-06"})
    assert response.status_code == 200
    assert len(response.json()) == 20


def test_book_and_fetch(client: TestClient) -> None:
    created = client.post("/appointments", json=payload())
    assert created.status_code == 201
    fetched = client.get(f"/appointments/{created.json()['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["pet_name"] == "Rex"


def test_double_booking_returns_conflict(client: TestClient) -> None:
    client.post("/appointments", json=payload())
    response = client.post("/appointments", json=payload())
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "slot_unavailable"


def test_outside_business_hours_returns_unprocessable(client: TestClient) -> None:
    response = client.post("/appointments", json=payload("2026-10-11T10:00:00"))
    assert response.status_code == 422


def test_reschedule_and_cancel(client: TestClient) -> None:
    appointment_id = client.post("/appointments", json=payload()).json()["id"]
    moved = client.patch(
        f"/appointments/{appointment_id}", json={"new_start": "2026-10-06T11:00:00"}
    )
    assert moved.status_code == 200
    assert moved.json()["start"] == "2026-10-06T11:00:00"
    cancelled = client.delete(f"/appointments/{appointment_id}")
    assert cancelled.json()["status"] == "cancelled"


def test_unknown_appointment_returns_not_found(client: TestClient) -> None:
    assert client.get("/appointments/missing").status_code == 404
