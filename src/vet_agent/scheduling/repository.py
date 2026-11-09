from datetime import date
from typing import Protocol

from vet_agent.scheduling.models import Appointment, AppointmentStatus


class SlotConflictError(Exception):
    pass


class AppointmentRepository(Protocol):
    def add(self, appointment: Appointment) -> None: ...

    def get(self, appointment_id: str) -> Appointment | None: ...

    def update(self, appointment: Appointment) -> None: ...

    def list_by_day(self, day: date) -> list[Appointment]: ...


class InMemoryAppointmentRepository:
    def __init__(self) -> None:
        self._items: dict[str, Appointment] = {}

    def add(self, appointment: Appointment) -> None:
        self._ensure_slot_free(appointment)
        self._items[appointment.id] = appointment

    def get(self, appointment_id: str) -> Appointment | None:
        return self._items.get(appointment_id)

    def update(self, appointment: Appointment) -> None:
        self._ensure_slot_free(appointment)
        self._items[appointment.id] = appointment

    def list_by_day(self, day: date) -> list[Appointment]:
        items = [item for item in self._items.values() if item.start.date() == day]
        return sorted(items, key=lambda item: item.start)

    def _ensure_slot_free(self, appointment: Appointment) -> None:
        if appointment.status != AppointmentStatus.SCHEDULED:
            return
        for item in self._items.values():
            if (
                item.id != appointment.id
                and item.status == AppointmentStatus.SCHEDULED
                and item.start == appointment.start
            ):
                raise SlotConflictError(appointment.start.isoformat())
