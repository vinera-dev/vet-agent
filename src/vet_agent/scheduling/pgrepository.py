from datetime import date, datetime, time, timedelta

import psycopg

from vet_agent.scheduling.models import Appointment, AppointmentStatus, ServiceType
from vet_agent.scheduling.repository import SlotConflictError

COLUMNS = "id, tutor_name, pet_name, service, start_at, status"


class PostgresAppointmentRepository:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def add(self, appointment: Appointment) -> None:
        self._write(
            f"INSERT INTO appointments ({COLUMNS}) VALUES (%s, %s, %s, %s, %s, %s)",
            (
                appointment.id,
                appointment.tutor_name,
                appointment.pet_name,
                appointment.service.value,
                appointment.start,
                appointment.status.value,
            ),
        )

    def get(self, appointment_id: str) -> Appointment | None:
        with psycopg.connect(self._database_url) as connection:
            row = connection.execute(
                f"SELECT {COLUMNS} FROM appointments WHERE id = %s", (appointment_id,)
            ).fetchone()
        return _to_appointment(row) if row else None

    def update(self, appointment: Appointment) -> None:
        self._write(
            "UPDATE appointments SET tutor_name = %s, pet_name = %s, service = %s, "
            "start_at = %s, status = %s WHERE id = %s",
            (
                appointment.tutor_name,
                appointment.pet_name,
                appointment.service.value,
                appointment.start,
                appointment.status.value,
                appointment.id,
            ),
        )

    def list_by_day(self, day: date) -> list[Appointment]:
        begin = datetime.combine(day, time.min)
        with psycopg.connect(self._database_url) as connection:
            rows = connection.execute(
                f"SELECT {COLUMNS} FROM appointments "
                "WHERE start_at >= %s AND start_at < %s ORDER BY start_at",
                (begin, begin + timedelta(days=1)),
            ).fetchall()
        return [_to_appointment(row) for row in rows]

    def _write(self, statement: str, parameters: tuple) -> None:
        try:
            with psycopg.connect(self._database_url) as connection:
                connection.execute(statement, parameters)
        except psycopg.errors.UniqueViolation as exc:
            raise SlotConflictError(str(exc)) from exc


def _to_appointment(row: tuple) -> Appointment:
    return Appointment(
        id=row[0],
        tutor_name=row[1],
        pet_name=row[2],
        service=ServiceType(row[3]),
        start=row[4],
        status=AppointmentStatus(row[5]),
    )
