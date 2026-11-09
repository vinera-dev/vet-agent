from vet_agent.config import get_settings
from vet_agent.llm import AzureOpenAIClient, LLMClient
from vet_agent.llm_local import LocalLLM


def get_llm() -> LLMClient:
    settings = get_settings()
    if settings.llm_backend == "local":
        return LocalLLM()
    return AzureOpenAIClient(settings)
