from datetime import date, datetime

import pytest

from vet_agent.scheduling.models import Appointment, AppointmentStatus, ServiceType
from vet_agent.scheduling.pgrepository import PostgresAppointmentRepository
from vet_agent.scheduling.repository import (
    AppointmentRepository,
    InMemoryAppointmentRepository,
    SlotConflictError,
)
from vet_agent.scheduling.service import SchedulingError, SchedulingService

NOW = datetime(2026, 10, 5, 9, 0)
TUESDAY = date(2026, 10, 6)


@pytest.fixture(params=["memory", "postgres"])
def repository(request: pytest.FixtureRequest) -> AppointmentRepository:
    if request.param == "memory":
        return InMemoryAppointmentRepository()
    return PostgresAppointmentRepository(request.getfixturevalue("postgres_url"))


def make(identifier: str, start: datetime, **changes) -> Appointment:
    values = {
        "id": identifier,
        "tutor_name": "Ana",
        "pet_name": "Rex",
        "service": ServiceType.CONSULTA,
        "start": start,
    }
    return Appointment(**{**values, **changes})


def test_add_then_get_returns_the_same_appointment(repository: AppointmentRepository) -> None:
    appointment = make("a1", datetime(2026, 10, 6, 10, 0), service=ServiceType.BANHO_TOSA)
    repository.add(appointment)
    assert repository.get("a1") == appointment


def test_get_unknown_id_returns_none(repository: AppointmentRepository) -> None:
    assert repository.get("missing") is None


def test_update_persists_new_values(repository: AppointmentRepository) -> None:
    appointment = make("a1", datetime(2026, 10, 6, 10, 0))
    repository.add(appointment)
    changed = appointment.model_copy(
        update={"start": datetime(2026, 10, 6, 11, 30), "status": AppointmentStatus.CANCELLED}
    )
    repository.update(changed)
    assert repository.get("a1") == changed


def test_list_by_day_returns_only_that_day_ordered_by_time(
    repository: AppointmentRepository,
) -> None:
    repository.add(make("late", datetime(2026, 10, 6, 17, 30)))
    repository.add(make("early", datetime(2026, 10, 6, 8, 0)))
    repository.add(make("last-minute", datetime(2026, 10, 6, 23, 59)))
    repository.add(make("next-day", datetime(2026, 10, 7, 0, 0)))
    repository.add(make("previous-day", datetime(2026, 10, 5, 23, 59)))
    listed = repository.list_by_day(TUESDAY)
    assert [item.id for item in listed] == ["early", "late", "last-minute"]


def test_list_by_day_includes_cancelled_appointments(repository: AppointmentRepository) -> None:
    repository.add(make("a1", datetime(2026, 10, 6, 10, 0), status=AppointmentStatus.CANCELLED))
    assert [item.id for item in repository.list_by_day(TUESDAY)] == ["a1"]


def test_two_active_appointments_cannot_share_a_slot(repository: AppointmentRepository) -> None:
    repository.add(make("a1", datetime(2026, 10, 6, 10, 0)))
    with pytest.raises(SlotConflictError):
        repository.add(make("a2", datetime(2026, 10, 6, 10, 0)))
    assert repository.get("a2") is None


def test_a_cancelled_slot_can_be_booked_again(repository: AppointmentRepository) -> None:
    start = datetime(2026, 10, 6, 10, 0)
    repository.add(make("a1", start, status=AppointmentStatus.CANCELLED))
    repository.add(make("a2", start))
    assert repository.get("a2") is not None


def test_moving_into_an_occupied_slot_is_rejected(repository: AppointmentRepository) -> None:
    repository.add(make("a1", datetime(2026, 10, 6, 10, 0)))
    repository.add(make("a2", datetime(2026, 10, 6, 11, 0)))
    with pytest.raises(SlotConflictError):
        repository.update(make("a2", datetime(2026, 10, 6, 10, 0)))
    assert repository.get("a2").start == datetime(2026, 10, 6, 11, 0)


def test_rewriting_an_appointment_without_moving_it_is_allowed(
    repository: AppointmentRepository,
) -> None:
    appointment = make("a1", datetime(2026, 10, 6, 10, 0))
    repository.add(appointment)
    repository.update(appointment.model_copy(update={"pet_name": "Thor"}))
    assert repository.get("a1").pet_name == "Thor"


def test_service_runs_the_booking_lifecycle_on_the_repository(
    repository: AppointmentRepository,
) -> None:
    service = SchedulingService(repository, now=lambda: NOW)
    start = datetime(2026, 10, 6, 10, 0)
    booked = service.book("Ana", "Rex", ServiceType.VACINA, start)
    assert start not in service.available_slots(TUESDAY)

    with pytest.raises(SchedulingError) as conflict:
        service.book("Bia", "Mel", ServiceType.CONSULTA, start)
    assert conflict.value.code == "slot_unavailable"

    moved = service.reschedule(booked.id, datetime(2026, 10, 6, 11, 0))
    assert moved.start == datetime(2026, 10, 6, 11, 0)
    assert start in service.available_slots(TUESDAY)

    service.cancel(booked.id)
    assert service.get(booked.id).status == AppointmentStatus.CANCELLED
    assert datetime(2026, 10, 6, 11, 0) in service.available_slots(TUESDAY)
