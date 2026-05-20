#!/usr/bin/env python3
"""
Load data from a SQL database into a CSV manifest for EDAgent analysis.

Supports any SQLAlchemy-compatible database (SQLite, PostgreSQL, MySQL, etc.).
Optionally joins multiple tables and applies column selection, filtering, and row limits.

Output: a CSV file ready for use as manifest_path in the EDAgent pipeline,
plus a sql_source_info.json with query metadata.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import write_json


def _redact_dsn(connection_string: str) -> str:
    """Remove password from connection string for safe logging."""
    return re.sub(r"(://[^:@/]*:)[^@]*(@)", r"\1***\2", connection_string)


def list_tables(engine) -> List[str]:
    """Return all table names in the connected database."""
    from sqlalchemy import inspect as sa_inspect
    return sa_inspect(engine).get_table_names()


def build_query(
    table: str,
    joins: List[Dict[str, str]],
    columns: Optional[List[str]],
    where: Optional[str],
    limit: int,
) -> str:
    """Build a SELECT query from parameters. No parameterisation needed — values are not user-supplied at runtime."""
    col_list = ", ".join(columns) if columns else "*"
    query = f"SELECT {col_list} FROM {table}"
    for join in joins:
        join_type = join.get("type", "left").upper()
        join_table = join["table"]
        on_clause = join["on"]
        query += f" {join_type} JOIN {join_table} ON {on_clause}"
    if where:
        query += f" WHERE {where}"
    if limit and limit > 0:
        query += f" LIMIT {limit}"
    return query


def load_sql(
    connection_string: str,
    table: str,
    joins: List[Dict[str, str]],
    columns: Optional[List[str]],
    where: Optional[str],
    limit: int,
    output_csv: Path,
    output_info: Path,
) -> pd.DataFrame:
    """Connect to the database, run the query, and write the manifest CSV."""
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        print("[ERROR] sqlalchemy is required: pip install sqlalchemy", file=sys.stderr)
        sys.exit(1)

    try:
        engine = create_engine(connection_string)
    except Exception as e:
        print(f"[ERROR] Failed to create engine: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        available_tables = list_tables(engine)
    except Exception as e:
        print(f"[ERROR] Failed to inspect database: {e}", file=sys.stderr)
        sys.exit(1)

    if table not in available_tables:
        print(
            f"[ERROR] Table '{table}' not found.\nAvailable tables: {available_tables}",
            file=sys.stderr,
        )
        sys.exit(1)

    for join in joins:
        jt = join.get("table", "")
        if jt and jt not in available_tables:
            print(
                f"[ERROR] Join table '{jt}' not found.\nAvailable tables: {available_tables}",
                file=sys.stderr,
            )
            sys.exit(1)

    query = build_query(table, joins, columns, where, limit)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
    except Exception as e:
        print(f"[ERROR] Query failed: {e}\nQuery: {query}", file=sys.stderr)
        sys.exit(1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    info: Dict[str, Any] = {
        "source_type": "sql",
        "connection_string": _redact_dsn(connection_string),
        "primary_table": table,
        "joins": joins,
        "where_clause": where or "",
        "columns_selected": columns or [],
        "row_limit": limit,
        "query": query,
        "output_csv": str(output_csv),
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "column_names": list(df.columns),
    }
    write_json(output_info, info)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load SQL table(s) into a CSV manifest for EDAgent."
    )
    parser.add_argument("connection_string", help="SQLAlchemy connection URL")
    parser.add_argument("--table", required=True, help="Primary table to query")
    parser.add_argument(
        "--join",
        action="append",
        dest="joins",
        default=[],
        metavar="JSON",
        help=(
            'JSON join spec, e.g. \'{"table":"t2","on":"t1.id=t2.fk","type":"left"}\'. '
            "Repeatable for multiple joins."
        ),
    )
    parser.add_argument("--where", default="", help="WHERE clause (omit the WHERE keyword)")
    parser.add_argument(
        "--columns",
        default="",
        help="Comma-separated column names to select; empty selects all columns",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum rows to load (0 = no limit)")
    parser.add_argument(
        "--output-csv",
        default="",
        help="Output CSV path; defaults to <output-dir>/sql_manifest.csv",
    )
    parser.add_argument("--output-dir", default="eda_outputs", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_csv = Path(args.output_csv) if args.output_csv else output_dir / "sql_manifest.csv"
    output_info = output_dir / "sql_source_info.json"

    joins: List[Dict[str, str]] = []
    for raw in args.joins:
        try:
            joins.append(json.loads(raw))
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid join JSON: {raw!r} — {e}", file=sys.stderr)
            sys.exit(1)

    columns = [c.strip() for c in args.columns.split(",") if c.strip()] if args.columns else None

    df = load_sql(
        connection_string=args.connection_string,
        table=args.table,
        joins=joins,
        columns=columns,
        where=args.where or None,
        limit=args.limit,
        output_csv=output_csv,
        output_info=output_info,
    )

    print(
        json.dumps(
            {
                "manifest_path": str(output_csv),
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "sql_source_info": str(output_info),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
