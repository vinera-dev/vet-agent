from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from vet_agent.scheduling.models import ServiceType
from vet_agent.scheduling.service import SchedulingError, SchedulingService


class ListSlotsArgs(BaseModel):
    day: date


class BookAppointmentArgs(BaseModel):
    tutor_name: str = Field(min_length=1, max_length=120)
    pet_name: str = Field(min_length=1, max_length=120)
    service: ServiceType
    start: datetime


class RescheduleAppointmentArgs(BaseModel):
    appointment_id: str = Field(min_length=1)
    new_start: datetime


class CancelAppointmentArgs(BaseModel):
    appointment_id: str = Field(min_length=1)


TOOL_ARGUMENTS: dict[str, type[BaseModel]] = {
    "list_available_slots": ListSlotsArgs,
    "book_appointment": BookAppointmentArgs,
    "reschedule_appointment": RescheduleAppointmentArgs,
    "cancel_appointment": CancelAppointmentArgs,
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "list_available_slots": "Lista os horários livres de um dia (formato AAAA-MM-DD).",
    "book_appointment": "Agenda uma consulta, vacina, banho e tosa ou exame.",
    "reschedule_appointment": "Altera o horário de um agendamento ativo.",
    "cancel_appointment": "Cancela um agendamento ativo.",
}


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": TOOL_DESCRIPTIONS[name],
                "parameters": model.model_json_schema(),
            },
        }
        for name, model in TOOL_ARGUMENTS.items()
    ]


def execute_tool(
    service: SchedulingService, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    model = TOOL_ARGUMENTS.get(name)
    if model is None:
        return _error("unknown_tool", f"Ferramenta desconhecida: {name}.")
    try:
        args = model.model_validate(arguments)
    except ValidationError as exc:
        return _error("invalid_arguments", str(exc))
    try:
        return {"ok": True, "result": _dispatch(service, name, args)}
    except SchedulingError as exc:
        return _error(exc.code, exc.message)


def _dispatch(service: SchedulingService, name: str, args: BaseModel) -> Any:
    if isinstance(args, ListSlotsArgs):
        return [slot.isoformat() for slot in service.available_slots(args.day)]
    if isinstance(args, BookAppointmentArgs):
        return service.book(args.tutor_name, args.pet_name, args.service, args.start).model_dump(
            mode="json"
        )
    if isinstance(args, RescheduleAppointmentArgs):
        return service.reschedule(args.appointment_id, args.new_start).model_dump(mode="json")
    if isinstance(args, CancelAppointmentArgs):
        return service.cancel(args.appointment_id).model_dump(mode="json")
    raise SchedulingError("unknown_tool", f"Ferramenta desconhecida: {name}.")


def _error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}
