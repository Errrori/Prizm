"""SQLite 连接器：用 stdlib sqlite3 只读读取单张表，避免 DuckDB 扩展联网。"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd

from prizm.datasource.base import BaseConnector
from prizm.engine.duckdb_engine import quote_identifier

_TABLE_QUERY = (
    "SELECT name FROM sqlite_master "
    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
    "ORDER BY name"
)


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def list_sqlite_tables(path: str | Path) -> tuple[str, ...]:
    """列出数据库中的用户表（排除 sqlite_* 内部表）。"""
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"数据文件不存在: {file}")
    try:
        with closing(_connect_readonly(file)) as conn:
            rows = conn.execute(_TABLE_QUERY).fetchall()
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"无法读取 SQLite 文件 {file.name}: {exc}") from exc
    return tuple(str(row[0]) for row in rows)


class SqliteConnector(BaseConnector):
    def __init__(
        self,
        path: str | Path,
        view_name: str | None = None,
        *,
        table: str,
        **read_sql_kwargs: Any,
    ) -> None:
        super().__init__(path, view_name or table)
        self._table = table
        self._read_kwargs = read_sql_kwargs

    @property
    def table(self) -> str:
        return self._table

    def _read(self) -> pd.DataFrame:
        # DEBT-021: 仅支持单表整表读入, 不支持视图、附加库与多表关联。
        tables = list_sqlite_tables(self.path)
        if not tables:
            raise ValueError(f"SQLite 文件没有可读取的数据表: {self.path.name}")
        if self._table not in tables:
            available = "、".join(tables)
            raise ValueError(
                f"数据表不存在: {self._table}（{self.path.name} 可用数据表: {available}）"
            )
        sql = f"SELECT * FROM {quote_identifier(self._table)}"
        try:
            with closing(_connect_readonly(self.path)) as conn:
                return pd.read_sql_query(sql, conn, **self._read_kwargs)
        except sqlite3.Error as exc:
            raise ValueError(f"读取 SQLite 数据表 {self._table} 失败: {exc}") from exc
