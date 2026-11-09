from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from vet_agent.config import Settings


def build_checkpointer(settings: Settings) -> BaseCheckpointSaver:
    if settings.session_backend == "postgres":
        return postgres_checkpointer(settings.database_url)
    return InMemorySaver()


def postgres_checkpointer(database_url: str) -> PostgresSaver:
    pool = ConnectionPool(
        database_url,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )
    saver = PostgresSaver(pool)
    saver.setup()
    return saver
