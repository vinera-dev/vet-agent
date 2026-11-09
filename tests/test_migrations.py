from pathlib import Path

import psycopg
import pytest

from vet_agent.db.migrate import apply_migrations

MIGRATIONS_DIRECTORY = Path(__file__).resolve().parent.parent / "migrations"


def test_migrations_run_from_scratch_and_are_recorded(empty_database_url: str) -> None:
    applied = apply_migrations(empty_database_url, MIGRATIONS_DIRECTORY)

    assert applied == [path.stem for path in sorted(MIGRATIONS_DIRECTORY.glob("*.sql"))]
    with psycopg.connect(empty_database_url) as connection:
        recorded = [row[0] for row in connection.execute("SELECT version FROM schema_migrations")]
        columns = {
            row[0]
            for row in connection.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'appointments'"
            )
        }
    assert recorded == applied
    assert columns == {"id", "tutor_name", "pet_name", "service", "start_at", "status"}


def test_running_migrations_twice_applies_nothing_new(empty_database_url: str) -> None:
    apply_migrations(empty_database_url, MIGRATIONS_DIRECTORY)
    assert apply_migrations(empty_database_url, MIGRATIONS_DIRECTORY) == []


def test_failed_migration_is_rolled_back(empty_database_url: str, tmp_path: Path) -> None:
    (tmp_path / "0001_ok.sql").write_text("CREATE TABLE first_table (id INT);")
    (tmp_path / "0002_broken.sql").write_text(
        "CREATE TABLE second_table (id INT); SELECT * FROM missing_table;"
    )
    with pytest.raises(psycopg.errors.UndefinedTable):
        apply_migrations(empty_database_url, tmp_path)

    with psycopg.connect(empty_database_url) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
        }
        recorded = [row[0] for row in connection.execute("SELECT version FROM schema_migrations")]
    assert "first_table" in tables
    assert "second_table" not in tables
    assert recorded == ["0001_ok"]
