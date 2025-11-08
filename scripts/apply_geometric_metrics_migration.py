#!/usr/bin/env python3
"""
Apply the geometric metrics migration using SQLAlchemy.

Requires DATABASE_URL environment variable pointing to the TimescaleDB/Postgres instance.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import text, create_engine


def main() -> int:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL environment variable is not set.", file=sys.stderr)
        return 1

    migration_path = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "infrastructure"
        / "database"
        / "migrations"
        / "20250214_add_geometric_metrics_events.sql"
    )

    if not migration_path.exists():
        print(f"Migration file not found: {migration_path}", file=sys.stderr)
        return 1

    migration_sql = migration_path.read_text(encoding="utf-8")

    engine = create_engine(database_url)
    with engine.connect() as connection:
        for statement in filter(None, (stmt.strip() for stmt in migration_sql.split(";"))):
            connection.execute(text(statement))
        connection.commit()

    print("Geometric metrics migration applied successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
