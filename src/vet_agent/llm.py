from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol

from openai import AzureOpenAI

from vet_agent.config import Settings


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int


_usage: ContextVar[TokenUsage | None] = ContextVar("llm_usage", default=None)


def reset_usage() -> None:
    _usage.set(None)


def last_usage() -> TokenUsage | None:
    return _usage.get()


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AzureOpenAIClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AzureOpenAI | None = None

    def _get_client(self) -> AzureOpenAI:
        if self._client is None:
            self._client = AzureOpenAI(
                azure_endpoint=self._settings.azure_openai_endpoint,
                api_key=self._settings.azure_openai_api_key,
                api_version=self._settings.azure_openai_api_version,
            )
        return self._client

    def complete(self, system: str, user: str) -> str:
        response = self._get_client().chat.completions.create(
            model=self._settings.azure_openai_deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if response.usage is not None:
            _usage.set(
                TokenUsage(response.usage.prompt_tokens, response.usage.completion_tokens)
            )
        return response.choices[0].message.content or ""
