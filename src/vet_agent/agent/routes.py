from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from vet_agent.agent.models import AgentReply
from vet_agent.agent.service import AgentService
from vet_agent.dependencies import get_llm
from vet_agent.rag.routes import get_faq_service
from vet_agent.scheduling.routes import get_scheduling_service

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=100)


@lru_cache
def get_agent_service() -> AgentService:
    llm = get_llm()
    return AgentService(
        llm=llm,
        faq=get_faq_service(llm),
        scheduling=get_scheduling_service(),
    )


AgentDep = Annotated[AgentService, Depends(get_agent_service)]


@router.post("/chat", response_model=AgentReply)
def chat(request: AgentChatRequest, service: AgentDep) -> AgentReply:
    return service.chat(request.message, request.session_id)
