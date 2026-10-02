from typing import Protocol

from openai import AzureOpenAI

from vet_agent.config import Settings


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
        return response.choices[0].message.content or ""
