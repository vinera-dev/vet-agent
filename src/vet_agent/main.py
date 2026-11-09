from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from vet_agent import __version__
from vet_agent.agent.routes import router as agent_router
from vet_agent.config import get_settings
from vet_agent.db.migrate import apply_migrations
from vet_agent.dependencies import get_llm
from vet_agent.llm import LLMClient
from vet_agent.observability import get_tracer
from vet_agent.rag.routes import router as faq_router
from vet_agent.scheduling.routes import router as scheduling_router

SYSTEM_PROMPT = (
    "Você é o assistente de uma clínica veterinária. Responda de forma clara e cordial. "
    "Não faça diagnósticos nem indique medicamentos; em caso de urgência, oriente o tutor "
    "a procurar atendimento presencial imediato."
)

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if settings.appointment_backend == "postgres":
        apply_migrations(settings.database_url, Path(settings.migrations_directory))
    yield
    get_tracer().flush()


app = FastAPI(title="vet-agent", version=__version__, lifespan=lifespan)
app.include_router(scheduling_router)
app.include_router(faq_router)
app.include_router(agent_router)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, llm: Annotated[LLMClient, Depends(get_llm)]) -> ChatResponse:
    return ChatResponse(reply=llm.complete(SYSTEM_PROMPT, request.message))
