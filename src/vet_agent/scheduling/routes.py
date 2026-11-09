from datetime import date, datetime
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from vet_agent.config import get_settings
from vet_agent.scheduling.models import Appointment, ServiceType
from vet_agent.scheduling.pgrepository import PostgresAppointmentRepository
from vet_agent.scheduling.repository import AppointmentRepository, InMemoryAppointmentRepository
from vet_agent.scheduling.service import SchedulingError, SchedulingService

STATUS_BY_ERROR_CODE = {
    "not_found": 404,
    "slot_unavailable": 409,
    "invalid_status": 409,
    "past_slot": 422,
    "outside_business_hours": 422,
}

router = APIRouter(prefix="/appointments", tags=["appointments"])


class BookRequest(BaseModel):
    tutor_name: str = Field(min_length=1, max_length=120)
    pet_name: str = Field(min_length=1, max_length=120)
    service: ServiceType
    start: datetime


class RescheduleRequest(BaseModel):
    new_start: datetime


def build_repository() -> AppointmentRepository:
    settings = get_settings()
    if settings.appointment_backend == "postgres":
        return PostgresAppointmentRepository(settings.database_url)
    return InMemoryAppointmentRepository()


@lru_cache
def get_scheduling_service() -> SchedulingService:
    return SchedulingService(build_repository())


SchedulingDep = Annotated[SchedulingService, Depends(get_scheduling_service)]


def _http_error(exc: SchedulingError) -> HTTPException:
    return HTTPException(
        status_code=STATUS_BY_ERROR_CODE.get(exc.code, 400),
        detail={"code": exc.code, "message": exc.message},
    )


@router.get("/slots", response_model=list[datetime])
def list_slots(day: date, service: SchedulingDep) -> list[datetime]:
    return service.available_slots(day)


@router.post("", response_model=Appointment, status_code=201)
def book(request: BookRequest, service: SchedulingDep) -> Appointment:
    try:
        return service.book(request.tutor_name, request.pet_name, request.service, request.start)
    except SchedulingError as exc:
        raise _http_error(exc) from exc


@router.get("/{appointment_id}", response_model=Appointment)
def get_appointment(appointment_id: str, service: SchedulingDep) -> Appointment:
    try:
        return service.get(appointment_id)
    except SchedulingError as exc:
        raise _http_error(exc) from exc


@router.patch("/{appointment_id}", response_model=Appointment)
def reschedule(
    appointment_id: str, request: RescheduleRequest, service: SchedulingDep
) -> Appointment:
    try:
        return service.reschedule(appointment_id, request.new_start)
    except SchedulingError as exc:
        raise _http_error(exc) from exc


@router.delete("/{appointment_id}", response_model=Appointment)
def cancel(appointment_id: str, service: SchedulingDep) -> Appointment:
    try:
        return service.cancel(appointment_id)
    except SchedulingError as exc:
        raise _http_error(exc) from exc
