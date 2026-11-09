from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_backend: Literal["azure", "local"] = "azure"
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_deployment: str = ""
    azure_openai_embedding_deployment: str = ""

    database_url: str = "postgresql://vet:vet@localhost:5432/vet_agent"

    embedding_backend: Literal["azure", "hashing"] = "hashing"
    embedding_dimensions: int = 1536
    vector_store_backend: Literal["memory", "pgvector"] = "memory"
    faq_directory: str = "data/faq"
    rag_top_k: int = 3
    rag_min_score: float = 0.1
    agent_rate_limit_requests: int = 20
    agent_rate_limit_window_seconds: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()
