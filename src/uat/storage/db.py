"""DuckDB connection + migrations.

Minute bars deliberately do NOT live here -- they go to date-partitioned Parquet
under data/parquet/minute/ and are read via read_parquet() globs, which keeps the
DB file small and makes the (re-fetchable) minute layer disposable.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd

from uat.config import load_settings, project_root

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def db_path() -> Path:
    p = load_settings().path("paths.duckdb")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def connect(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    con = duckdb.connect(str(db_path()), read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def migrate() -> list[int]:
    """Apply any migration files not yet recorded in sys.schema_version."""
    applied: list[int] = []
    with connect() as con:
        for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
            version = int(sql_file.name.split("_", 1)[0])
            con.execute(sql_file.read_text(encoding="utf-8"))
            con.execute(
                "INSERT INTO sys.schema_version VALUES (?, ?) ON CONFLICT DO NOTHING",
                [version, datetime.now()],
            )
            applied.append(version)
    return applied


def upsert_df(table: str, df: pd.DataFrame, key_cols: list[str]) -> int:
    """Insert a DataFrame, replacing rows that collide on the primary key.

    DuckDB's ON CONFLICT needs an explicit conflict target, and re-ingesting a day
    of bars must be idempotent, so we delete the overlapping keys first.
    """
    if df.empty:
        return 0
    with connect() as con:
        con.register("_incoming", df)
        key_expr = ", ".join(key_cols)
        con.execute(
            f"DELETE FROM {table} WHERE ({key_expr}) IN (SELECT {key_expr} FROM _incoming)"
        )
        cols = ", ".join(df.columns)
        con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _incoming")
        con.unregister("_incoming")
    return len(df)


def query(sql: str, params: list | None = None) -> pd.DataFrame:
    with connect(read_only=True) as con:
        return con.execute(sql, params or []).df()


def table_counts() -> pd.DataFrame:
    with connect(read_only=True) as con:
        tables = con.execute(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_schema NOT IN ('information_schema','pg_catalog') ORDER BY 1,2"
        ).fetchall()
        rows = []
        for schema, name in tables:
            n = con.execute(f'SELECT COUNT(*) FROM "{schema}"."{name}"').fetchone()[0]
            rows.append({"table": f"{schema}.{name}", "rows": n})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="DuckDB schema management")
    ap.add_argument("--migrate", action="store_true", help="apply migrations")
    ap.add_argument("--status", action="store_true", help="show table row counts")
    args = ap.parse_args()
    if args.migrate:
        versions = migrate()
        print(f"applied migrations: {versions}  -> {db_path()}")
    if args.status or not args.migrate:
        print(table_counts().to_string(index=False))


if __name__ == "__main__":
    main()
