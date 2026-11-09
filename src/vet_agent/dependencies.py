from vet_agent.config import get_settings
from vet_agent.llm import AzureOpenAIClient, LLMClient
from vet_agent.llm_local import LocalLLM
from vet_agent.observability import TracedLLM, get_tracer


def get_llm() -> LLMClient:
    settings = get_settings()
    if settings.llm_backend == "local":
        llm: LLMClient = LocalLLM()
        model = "local-rule-based"
    else:
        llm = AzureOpenAIClient(settings)
        model = settings.azure_openai_deployment
    tracer = get_tracer()
    return TracedLLM(llm, tracer, model) if tracer.enabled else llm
