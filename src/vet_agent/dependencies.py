from vet_agent.config import get_settings
from vet_agent.llm import AzureOpenAIClient, LLMClient


def get_llm() -> LLMClient:
    return AzureOpenAIClient(get_settings())
