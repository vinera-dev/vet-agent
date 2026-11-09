import json
import re
from datetime import date, datetime, timedelta

from vet_agent.agent.messages import SMALLTALK_MESSAGE

TURN_PATTERN = re.compile(r"^(Tutor|Assistente): ", re.MULTILINE)
NOW_PATTERN = re.compile(r"Agora é (\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})")
UUID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
ISO_PATTERN = re.compile(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})")
DATE_PATTERN = re.compile(r"(\d{1,2})/(\d{1,2})")
TIME_PATTERN = re.compile(r"(?:às|as)?\s*(\d{1,2})\s*(?:h|:)\s*(\d{2})?")
DAY_ONLY_PATTERN = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
NAME_PATTERN = re.compile(r"(?i:meu nome é|me chamo|sou o|sou a)\s+([A-ZÀ-Ú][\wÀ-ú]+)")
PET_PATTERN = re.compile(
    r"(?i:pet|cachorro|cachorra|gato|gata|cão)\s+(?:(?i:se chama|chama-se|é o|é a)\s+)?"
    r"([A-ZÀ-Ú][\wÀ-ú]+)"
)
CONFIRMATION_PATTERN = re.compile(r"^\W*(sim|confirmo|confirmado|pode|isso|ok)\b", re.IGNORECASE)
SERVICES = {
    "banho_tosa": ("banho", "tosa"),
    "vacina": ("vacina",),
    "exame": ("exame",),
    "consulta": ("consulta",),
}
COMPLETION_MARKERS = (
    "Agendamento confirmado",
    "Agendamento cancelado",
    "Agendamento remarcado",
    "Horários livres",
)
MEDICAL_WORDS = ("diagnóstico", "remédio", "medicamento", "dose", "tratamento", "posso dar")
SCHEDULING_WORDS = (
    "agendar",
    "marcar",
    "remarcar",
    "reagendar",
    "cancelar",
    "horários livres",
    "horários disponíveis",
    "vaga",
)
FAQ_WORDS = ("preço", "valor", "quanto", "funciona", "funcionamento", "abre", "fecha", "endereço")
FLOW_CUES = ("agend", "confirma", "código", "qual o nome", "qual serviço", "para qual")


class LocalLLM:
    def complete(self, system: str, user: str) -> str:
        if system.startswith("Classifique"):
            return json.dumps({"intent": self._intent(user)})
        if system.startswith("Você agenda consultas"):
            return json.dumps(self._plan(system, user))
        if "<contexto>" in user:
            return self._faq_answer(user)
        return SMALLTALK_MESSAGE

    def _intent(self, transcript: str) -> str:
        turns = _turns(transcript)
        last = turns[-1][1].lower() if turns else ""
        previous = turns[-2][1].lower() if len(turns) > 1 else ""
        if any(word in last for word in MEDICAL_WORDS):
            return "medical_advice"
        if any(word in last for word in SCHEDULING_WORDS):
            return "scheduling"
        if any(cue in previous for cue in FLOW_CUES) and "?" not in last:
            return "scheduling"
        if "?" in last or any(word in last for word in FAQ_WORDS):
            return "faq"
        return "smalltalk"

    def _faq_answer(self, user: str) -> str:
        context = user.split("<contexto>", 1)[1].split("</contexto>", 1)[0].strip()
        passage = context.split("\n\n", 1)[0]
        return "Segundo os materiais da clínica: " + " ".join(passage.split())

    def _plan(self, system: str, user: str) -> dict:
        now = _now(system)
        transcript, _, results = user.partition("\n\nResultados das ferramentas nesta rodada:\n")
        if results:
            return {"action": "reply", "message": _summarize(json.loads(results)[-1])}
        turns = _turns(transcript)
        flow = _current_flow(turns)
        text = " ".join(content for role, content in flow if role == "Tutor")
        lowered = text.lower()
        confirmed = _confirmed(turns)
        if any(word in lowered for word in ("cancelar", "cancela")):
            return _cancel(text, confirmed)
        if any(word in lowered for word in ("remarcar", "reagendar")):
            return _reschedule(text, now, confirmed)
        if "horários livres" in lowered or "horários disponíveis" in lowered:
            return _list_slots(text, now)
        return _book(text, now, confirmed)


def _turns(transcript: str) -> list[tuple[str, str]]:
    parts = TURN_PATTERN.split(transcript)
    return [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]


def _now(system: str) -> datetime:
    match = NOW_PATTERN.search(system)
    if match is None:
        return datetime.now()
    return datetime(*(int(group) for group in match.groups()))


def _current_flow(turns: list[tuple[str, str]]) -> list[tuple[str, str]]:
    start = 0
    for index, (role, content) in enumerate(turns):
        if role == "Assistente" and any(marker in content for marker in COMPLETION_MARKERS):
            start = index + 1
    return turns[start:]


def _confirmed(turns: list[tuple[str, str]]) -> bool:
    if len(turns) < 2:
        return False
    (previous_role, previous), (last_role, last) = turns[-2], turns[-1]
    return (
        previous_role == "Assistente"
        and "Confirma" in previous
        and last_role == "Tutor"
        and CONFIRMATION_PATTERN.match(last) is not None
    )


def _ask(message: str) -> dict:
    return {"action": "reply", "message": message}


def _tool(name: str, **arguments: str) -> dict:
    return {"action": "tool", "tool": name, "arguments": arguments}


def _service(text: str) -> str | None:
    lowered = text.lower()
    for service, words in SERVICES.items():
        if any(word in lowered for word in words):
            return service
    return None


def _moment(text: str, now: datetime) -> datetime | None:
    iso = ISO_PATTERN.findall(text)
    if iso:
        return datetime(*(int(group) for group in iso[-1]))
    lowered = text.lower()
    day: date | None = None
    dates = DATE_PATTERN.findall(text)
    if dates:
        month, number = int(dates[-1][1]), int(dates[-1][0])
        day = date(now.year, month, number)
        if day < now.date():
            day = date(now.year + 1, month, number)
    elif "amanhã" in lowered:
        day = now.date() + timedelta(days=1)
    elif "hoje" in lowered:
        day = now.date()
    times = TIME_PATTERN.findall(re.sub(r"\d{1,2}/\d{1,2}", " ", text))
    if day is None or not times:
        return None
    hour, minute = times[-1]
    return datetime.combine(day, datetime.min.time()).replace(
        hour=int(hour), minute=int(minute or 0)
    )


def _day(text: str, now: datetime) -> date | None:
    match = DAY_ONLY_PATTERN.findall(text)
    if match:
        return date(*(int(group) for group in match[-1]))
    lowered = text.lower()
    dates = DATE_PATTERN.findall(text)
    if dates:
        return date(now.year, int(dates[-1][1]), int(dates[-1][0]))
    if "amanhã" in lowered:
        return now.date() + timedelta(days=1)
    if "hoje" in lowered:
        return now.date()
    return None


def _list_slots(text: str, now: datetime) -> dict:
    day = _day(text, now)
    if day is None:
        return _ask("Para qual dia você quer ver os horários livres?")
    return _tool("list_available_slots", day=day.isoformat())


def _book(text: str, now: datetime, confirmed: bool) -> dict:
    name = _last(NAME_PATTERN, text)
    pet = _last(PET_PATTERN, text)
    service = _service(text)
    moment = _moment(text, now)
    if name is None:
        return _ask("Claro! Qual o nome do tutor?")
    if pet is None:
        return _ask("Qual o nome do pet?")
    if service is None:
        return _ask("Qual serviço você deseja: consulta, vacina, banho e tosa ou exame?")
    if moment is None:
        return _ask("Para qual dia e horário? Por exemplo: dia 10/11 às 10h.")
    if not confirmed:
        return _ask(
            f"Confirma o agendamento de {service.replace('_', ' ')} para {pet}, "
            f"tutor {name}, em {moment:%d/%m/%Y} às {moment:%H:%M}? (sim/não)"
        )
    return _tool(
        "book_appointment",
        tutor_name=name,
        pet_name=pet,
        service=service,
        start=moment.isoformat(),
    )


def _cancel(text: str, confirmed: bool) -> dict:
    ids = UUID_PATTERN.findall(text)
    if not ids:
        return _ask("Qual o código do agendamento que deseja cancelar?")
    if not confirmed:
        return _ask(f"Confirma o cancelamento do agendamento {ids[-1]}? (sim/não)")
    return _tool("cancel_appointment", appointment_id=ids[-1])


def _reschedule(text: str, now: datetime, confirmed: bool) -> dict:
    ids = UUID_PATTERN.findall(text)
    if not ids:
        return _ask("Qual o código do agendamento que deseja remarcar?")
    moment = _moment(re.sub(UUID_PATTERN, " ", text), now)
    if moment is None:
        return _ask("Para qual novo dia e horário? Por exemplo: dia 10/11 às 10h.")
    if not confirmed:
        return _ask(
            f"Confirma a remarcação do agendamento {ids[-1]} para "
            f"{moment:%d/%m/%Y} às {moment:%H:%M}? (sim/não)"
        )
    return _tool("reschedule_appointment", appointment_id=ids[-1], new_start=moment.isoformat())


def _last(pattern: re.Pattern[str], text: str) -> str | None:
    matches = pattern.findall(text)
    return matches[-1] if matches else None


def _summarize(record: dict) -> str:
    result = record["result"]
    if not result["ok"]:
        return f"Não consegui concluir: {result['error']['message']} Quer escolher outro horário?"
    data = result["result"]
    name = record["name"]
    if name == "list_available_slots":
        slots = [datetime.fromisoformat(slot) for slot in data]
        if not slots:
            return "Horários livres: nenhum nesse dia."
        return "Horários livres: " + ", ".join(f"{slot:%H:%M}" for slot in slots[:8]) + "."
    start = datetime.fromisoformat(data["start"])
    when = f"{start:%d/%m/%Y} às {start:%H:%M}"
    if name == "book_appointment":
        return f"Agendamento confirmado. Código: {data['id']}. {data['pet_name']}, {when}."
    if name == "reschedule_appointment":
        return f"Agendamento remarcado para {when}. Código: {data['id']}."
    return f"Agendamento cancelado. Código: {data['id']}."
