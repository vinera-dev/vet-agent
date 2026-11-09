from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from vet_agent.agent.routes import get_agent_service
from vet_agent.main import app
from vet_agent.ratelimit import SlidingWindowLimiter, get_rate_limiter


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class StubAgent:
    def chat(self, message: str, session_id: str | None):
        from vet_agent.agent.models import AgentReply

        return AgentReply(reply="ok", intent="faq", session_id=session_id or "s", escalated=False)


@pytest.fixture
def clock() -> Clock:
    return Clock()


def test_requests_within_the_limit_are_allowed(clock):
    limiter = SlidingWindowLimiter(3, 60, clock)

    assert [limiter.check("a") for _ in range(3)] == [None, None, None]


def test_request_over_the_limit_returns_the_wait_time(clock):
    limiter = SlidingWindowLimiter(2, 60, clock)
    limiter.check("a")
    clock.now += 10
    limiter.check("a")
    clock.now += 5

    assert limiter.check("a") == pytest.approx(45.0)


def test_window_slides_instead_of_resetting(clock):
    limiter = SlidingWindowLimiter(2, 60, clock)
    limiter.check("a")
    clock.now += 30
    limiter.check("a")
    clock.now += 31

    assert limiter.check("a") is None
    assert limiter.check("a") is not None


def test_clients_have_independent_budgets(clock):
    limiter = SlidingWindowLimiter(1, 60, clock)

    assert limiter.check("a") is None
    assert limiter.check("a") is not None
    assert limiter.check("b") is None


def test_idle_clients_are_forgotten(clock):
    limiter = SlidingWindowLimiter(1, 60, clock)
    limiter.check("a")
    clock.now += 61
    limiter.check("b")

    assert list(limiter._hits) == ["b"]


@pytest.fixture
def client(clock) -> Iterator[TestClient]:
    limiter = SlidingWindowLimiter(2, 60, clock)
    app.dependency_overrides[get_agent_service] = StubAgent
    app.dependency_overrides[get_rate_limiter] = lambda: limiter
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_chat_endpoint_returns_429_with_retry_after_over_the_limit(client):
    statuses = [client.post("/agent/chat", json={"message": "oi"}).status_code for _ in range(3)]
    blocked = client.post("/agent/chat", json={"message": "oi"})

    assert statuses == [200, 200, 429]
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1


def test_chat_is_available_again_after_the_window(client, clock):
    for _ in range(2):
        client.post("/agent/chat", json={"message": "oi"})
    assert client.post("/agent/chat", json={"message": "oi"}).status_code == 429

    clock.now += 61

    assert client.post("/agent/chat", json={"message": "oi"}).status_code == 200


def test_other_endpoints_are_not_limited(client):
    for _ in range(5):
        client.post("/agent/chat", json={"message": "oi"})

    assert client.get("/health").status_code == 200
