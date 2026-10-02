from datetime import date, datetime

import pytest

from vet_agent.scheduling.models import AppointmentStatus, ServiceType
from vet_agent.scheduling.repository import InMemoryAppointmentRepository
from vet_agent.scheduling.service import SchedulingError, SchedulingService

NOW = datetime(2026, 10, 5, 9, 0)
TUESDAY = date(2026, 10, 6)
SUNDAY = date(2026, 10, 11)


@pytest.fixture
def service() -> SchedulingService:
    return SchedulingService(InMemoryAppointmentRepository(), now=lambda: NOW)


def book(service: SchedulingService, start: datetime):
    return service.book("Ana", "Rex", ServiceType.CONSULTA, start)


def test_available_slots_cover_business_hours(service: SchedulingService) -> None:
    slots = service.available_slots(TUESDAY)
    assert len(slots) == 20
    assert slots[0] == datetime(2026, 10, 6, 8, 0)
    assert slots[-1] == datetime(2026, 10, 6, 17, 30)


def test_no_slots_on_sunday(service: SchedulingService) -> None:
    assert service.available_slots(SUNDAY) == []


def test_past_slots_are_not_offered(service: SchedulingService) -> None:
    slots = service.available_slots(date(2026, 10, 5))
    assert slots[0] == datetime(2026, 10, 5, 9, 30)


def test_booked_slot_is_removed_from_availability(service: SchedulingService) -> None:
    start = datetime(2026, 10, 6, 10, 0)
    book(service, start)
    assert start not in service.available_slots(TUESDAY)


def test_book_returns_scheduled_appointment(service: SchedulingService) -> None:
    appointment = book(service, datetime(2026, 10, 6, 10, 0))
    assert appointment.status == AppointmentStatus.SCHEDULED
    assert service.get(appointment.id) == appointment


def test_double_booking_is_rejected(service: SchedulingService) -> None:
    start = datetime(2026, 10, 6, 10, 0)
    book(service, start)
    with pytest.raises(SchedulingError) as error:
        book(service, start)
    assert error.value.code == "slot_unavailable"


def test_booking_in_the_past_is_rejected(service: SchedulingService) -> None:
    with pytest.raises(SchedulingError) as error:
        book(service, datetime(2026, 10, 5, 8, 0))
    assert error.value.code == "past_slot"


@pytest.mark.parametrize(
    "start",
    [
        datetime(2026, 10, 6, 7, 30),
        datetime(2026, 10, 6, 18, 0),
        datetime(2026, 10, 6, 10, 15),
        datetime(2026, 10, 11, 10, 0),
    ],
)
def test_booking_outside_business_hours_is_rejected(
    service: SchedulingService, start: datetime
) -> None:
    with pytest.raises(SchedulingError) as error:
        book(service, start)
    assert error.value.code == "outside_business_hours"


def test_reschedule_moves_the_appointment_and_frees_the_old_slot(
    service: SchedulingService,
) -> None:
    old_start = datetime(2026, 10, 6, 10, 0)
    new_start = datetime(2026, 10, 6, 11, 0)
    appointment = book(service, old_start)
    updated = service.reschedule(appointment.id, new_start)
    assert updated.start == new_start
    slots = service.available_slots(TUESDAY)
    assert old_start in slots
    assert new_start not in slots


def test_reschedule_to_taken_slot_is_rejected(service: SchedulingService) -> None:
    first = book(service, datetime(2026, 10, 6, 10, 0))
    book(service, datetime(2026, 10, 6, 11, 0))
    with pytest.raises(SchedulingError) as error:
        service.reschedule(first.id, datetime(2026, 10, 6, 11, 0))
    assert error.value.code == "slot_unavailable"


def test_cancel_frees_the_slot(service: SchedulingService) -> None:
    start = datetime(2026, 10, 6, 10, 0)
    appointment = book(service, start)
    cancelled = service.cancel(appointment.id)
    assert cancelled.status == AppointmentStatus.CANCELLED
    assert start in service.available_slots(TUESDAY)


def test_cancelled_appointment_cannot_be_changed(service: SchedulingService) -> None:
    appointment = book(service, datetime(2026, 10, 6, 10, 0))
    service.cancel(appointment.id)
    with pytest.raises(SchedulingError) as cancel_error:
        service.cancel(appointment.id)
    with pytest.raises(SchedulingError) as reschedule_error:
        service.reschedule(appointment.id, datetime(2026, 10, 6, 11, 0))
    assert cancel_error.value.code == "invalid_status"
    assert reschedule_error.value.code == "invalid_status"


def test_unknown_appointment_is_not_found(service: SchedulingService) -> None:
    with pytest.raises(SchedulingError) as error:
        service.get("missing")
    assert error.value.code == "not_found"
