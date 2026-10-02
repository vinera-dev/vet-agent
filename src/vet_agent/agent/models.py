from typing import Any

from pydantic import BaseModel, Field

from vet_agent.rag.models import Source


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class AgentReply(BaseModel):
    session_id: str
    reply: str
    intent: str
    escalated: bool
    grounded: bool | None = None
    sources: list[Source] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
