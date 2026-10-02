from fastapi.testclient import TestClient

from vet_agent.main import SYSTEM_PROMPT, app, get_llm


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return "Atendemos de segunda a sábado."


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_uses_llm_with_system_prompt() -> None:
    fake = FakeLLM()
    app.dependency_overrides[get_llm] = lambda: fake
    try:
        client = TestClient(app)
        response = client.post("/chat", json={"message": "Qual o horário?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"reply": "Atendemos de segunda a sábado."}
    assert fake.calls == [(SYSTEM_PROMPT, "Qual o horário?")]


def test_chat_rejects_empty_message() -> None:
    client = TestClient(app)
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422
