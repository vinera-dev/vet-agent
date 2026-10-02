from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from vet_agent.config import get_settings
from vet_agent.dependencies import get_llm
from vet_agent.llm import LLMClient
from vet_agent.rag.embeddings import Embedder, build_embedder
from vet_agent.rag.models import Answer
from vet_agent.rag.pgstore import PgVectorStore
from vet_agent.rag.service import FaqService
from vet_agent.rag.store import InMemoryVectorStore, VectorStore

router = APIRouter(prefix="/faq", tags=["faq"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class IngestResponse(BaseModel):
    chunks: int


@lru_cache
def _build_store() -> VectorStore:
    settings = get_settings()
    if settings.vector_store_backend == "pgvector":
        return PgVectorStore(settings.database_url, settings.embedding_dimensions)
    return InMemoryVectorStore()


@lru_cache
def _build_embedder() -> Embedder:
    return build_embedder(get_settings())


def get_faq_service(llm: Annotated[LLMClient, Depends(get_llm)]) -> FaqService:
    settings = get_settings()
    return FaqService(
        embedder=_build_embedder(),
        store=_build_store(),
        llm=llm,
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
    )


FaqDep = Annotated[FaqService, Depends(get_faq_service)]


@router.post("/ingest", response_model=IngestResponse)
def ingest(service: FaqDep) -> IngestResponse:
    return IngestResponse(chunks=service.ingest(Path(get_settings().faq_directory)))


@router.post("/ask", response_model=Answer)
def ask(request: AskRequest, service: FaqDep) -> Answer:
    return service.ask(request.question)
