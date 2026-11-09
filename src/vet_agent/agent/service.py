from collections.abc import Callable
from datetime import datetime
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from vet_agent.agent.graph import build_agent_graph, initial_turn_state
from vet_agent.agent.models import AgentReply, ToolCall
from vet_agent.llm import LLMClient
from vet_agent.observability import Tracer
from vet_agent.rag.models import Source
from vet_agent.rag.service import FaqService
from vet_agent.scheduling.service import SchedulingService


class AgentService:
    def __init__(
        self,
        llm: LLMClient,
        faq: FaqService,
        scheduling: SchedulingService,
        now: Callable[[], datetime] = datetime.now,
        checkpointer: BaseCheckpointSaver | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._tracer = tracer or Tracer()
        self._graph = build_agent_graph(
            llm, faq, scheduling, now, checkpointer or InMemorySaver(), self._tracer
        )

    def chat(self, message: str, session_id: str | None = None) -> AgentReply:
        session = session_id or str(uuid4())
        config = {"configurable": {"thread_id": session}}
        with self._tracer.session(session), self._tracer.observe("agent.chat") as root:
            state = self._graph.invoke(initial_turn_state(message), config)
            if root is not None:
                root.update(
                    metadata={"intent": state["intent"], "escalated": state["escalated"]}
                )
        return AgentReply(
            session_id=session,
            reply=state["reply"],
            intent=state["intent"],
            escalated=state["escalated"],
            grounded=state["grounded"],
            sources=[Source(**source) for source in state["sources"]],
            tool_calls=[ToolCall(**call) for call in state["tool_calls"]],
        )
