from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ServiceType(StrEnum):
    CONSULTA = "consulta"
    VACINA = "vacina"
    BANHO_TOSA = "banho_tosa"
    EXAME = "exame"


class AppointmentStatus(StrEnum):
    SCHEDULED = "scheduled"
    CANCELLED = "cancelled"


class Appointment(BaseModel):
    id: str
    tutor_name: str = Field(min_length=1, max_length=120)
    pet_name: str = Field(min_length=1, max_length=120)
    service: ServiceType
    start: datetime
    status: AppointmentStatus = AppointmentStatus.SCHEDULED
