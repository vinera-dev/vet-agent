import sys
from pathlib import Path

import psycopg

from vet_agent.config import get_settings

LOCK_KEY = 7_204_001

VERSION_TABLE = (
    "CREATE TABLE IF NOT EXISTS schema_migrations ("
    "version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
)


def apply_migrations(database_url: str, directory: Path) -> list[str]:
    scripts = sorted(directory.glob("*.sql"))
    applied: list[str] = []
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute("SELECT pg_advisory_lock(%s)", (LOCK_KEY,))
        try:
            connection.execute(VERSION_TABLE)
            done = {row[0] for row in connection.execute("SELECT version FROM schema_migrations")}
            for script in scripts:
                if script.stem in done:
                    continue
                with connection.transaction():
                    connection.execute(script.read_text(encoding="utf-8"))
                    connection.execute(
                        "INSERT INTO schema_migrations (version) VALUES (%s)", (script.stem,)
                    )
                applied.append(script.stem)
        finally:
            connection.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
    return applied


def main() -> int:
    settings = get_settings()
    applied = apply_migrations(settings.database_url, Path(settings.migrations_directory))
    print(f"applied {len(applied)} migration(s): {', '.join(applied) or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
