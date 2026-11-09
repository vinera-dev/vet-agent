import os
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import psycopg
import pytest

from vet_agent.db.migrate import apply_migrations

MIGRATIONS_DIRECTORY = Path(__file__).resolve().parent.parent / "migrations"


def _base_url() -> str | None:
    return os.environ.get("POSTGRES_TEST_URL") or os.environ.get("PGVECTOR_TEST_URL")


@pytest.fixture
def empty_database_url() -> Iterator[str]:
    base = _base_url()
    if base is None:
        pytest.skip("POSTGRES_TEST_URL não definida")
    name = f"vet_test_{uuid4().hex[:12]}"
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    parts = urlsplit(base)
    url = urlunsplit(parts._replace(path=f"/{name}"))
    try:
        yield url
    finally:
        with psycopg.connect(base, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture
def postgres_url(empty_database_url: str) -> str:
    apply_migrations(empty_database_url, MIGRATIONS_DIRECTORY)
    return empty_database_url
