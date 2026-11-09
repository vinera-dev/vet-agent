import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from typing import Any

from langfuse import Langfuse, propagate_attributes

from vet_agent.config import get_settings
from vet_agent.llm import LLMClient, last_usage, reset_usage

SAFE_NODE_FIELDS = ("intent", "escalated", "grounded", "steps", "blocked")


class Tracer:
    def __init__(self, client: Langfuse | None = None) -> None:
        self._client = client

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def observe(self, name: str, as_type: str = "span", **fields: Any) -> Iterator[Any]:
        if self._client is None:
            yield None
            return
        with self._client.start_as_current_observation(
            name=name, as_type=as_type, **fields
        ) as observation:
            yield observation

    def session(self, session_id: str) -> Any:
        if self._client is None:
            return nullcontext()
        return propagate_attributes(session_id=session_id, trace_name="agent.chat")

    def node(self, name: str) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., Any]]:
        def decorate(function: Callable[..., dict[str, Any]]) -> Callable[..., Any]:
            @functools.wraps(function)
            def run(state: Any) -> dict[str, Any]:
                if self._client is None:
                    return function(state)
                with self.observe(name) as observation:
                    update = function(state)
                    observation.update(
                        metadata={key: update[key] for key in SAFE_NODE_FIELDS if key in update}
                    )
                    return update

            return run

        return decorate

    def flush(self) -> None:
        if self._client is not None:
            self._client.flush()


class TracedLLM:
    def __init__(self, inner: LLMClient, tracer: Tracer, model: str) -> None:
        self._inner = inner
        self._tracer = tracer
        self._model = model

    def complete(self, system: str, user: str) -> str:
        with self._tracer.observe(
            "llm.complete",
            as_type="generation",
            model=self._model,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        ) as generation:
            reset_usage()
            output = self._inner.complete(system, user)
            usage = last_usage()
            fields: dict[str, Any] = {"output": output}
            if usage is not None:
                fields["usage_details"] = {
                    "input": usage.prompt_tokens,
                    "output": usage.completion_tokens,
                }
            if generation is not None:
                generation.update(**fields)
            return output


@lru_cache
def get_tracer() -> Tracer:
    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return Tracer()
    return Tracer(
        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    )
