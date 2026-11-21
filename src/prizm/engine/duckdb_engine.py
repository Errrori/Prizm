"""DuckDB 只读执行引擎：注册 DataFrame 视图 + 限额执行 SQL。"""

from __future__ import annotations

import re
import threading
from types import TracebackType
from typing import Any

import duckdb
import pandas as pd

_MEMORY_LIMIT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:B|[KMGT]i?B)", re.IGNORECASE)

_FORBIDDEN_KEYWORDS = frozenset(
    {
        "alter",
        "attach",
        "call",
        "checkpoint",
        "copy",
        "create",
        "delete",
        "detach",
        "drop",
        "export",
        "grant",
        "import",
        "insert",
        "install",
        "load",
        "merge",
        "pragma",
        "revoke",
        "set",
        "truncate",
        "update",
        "vacuum",
    }
)

# 即使 enable_external_access=false, 也显式拦截可能触碰文件/网络的表函数。
_FORBIDDEN_FUNCTIONS = frozenset(
    {
        "delta_scan",
        "glob",
        "httpfs",
        "iceberg_scan",
        "mysql_scan",
        "parquet_scan",
        "postgres_scan",
        "read_csv",
        "read_csv_auto",
        "read_json",
        "read_json_auto",
        "read_ndjson",
        "read_ndjson_auto",
        "read_parquet",
        "sqlite_scan",
    }
)


class SQLValidationError(ValueError):
    """SQL 未通过只读校验。"""


class QueryTimeoutError(RuntimeError):
    """查询超过配置的超时时间。"""


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def ensure_read_only_sql(sql: str) -> str:
    """校验只允许单条 SELECT / WITH 查询，返回去掉尾部分号的 SQL。"""
    cleaned = sql.strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()
    if not cleaned:
        raise SQLValidationError("SQL 不能为空")
    if ";" in cleaned:
        raise SQLValidationError("只允许执行单条 SQL 语句")

    lowered = cleaned.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise SQLValidationError("只允许 SELECT / WITH 查询")

    for keyword in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", lowered):
            raise SQLValidationError(f"禁止的 SQL 关键字: {keyword.upper()}")
    for function in _FORBIDDEN_FUNCTIONS:
        if re.search(rf"\b{function}\s*\(", lowered):
            raise SQLValidationError(f"禁止的 SQL 函数: {function}")
    return cleaned


class DuckDBEngine:
    """内存 DuckDB 连接，注册 DataFrame 为视图后执行只读 SQL。"""

    def __init__(
        self,
        *,
        memory_limit: str = "1GB",
        threads: int = 2,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not _MEMORY_LIMIT_RE.fullmatch(memory_limit.strip()):
            raise ValueError(f"非法的 memory_limit: {memory_limit!r}")
        if threads < 1:
            raise ValueError("threads 必须 >= 1")
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds 不能为负数")

        self._timeout_seconds = timeout_seconds
        self._conn = duckdb.connect(":memory:")
        self._conn.execute(f"SET memory_limit='{memory_limit.strip()}'")
        self._conn.execute(f"SET threads={int(threads)}")
        # 安全不变量: 禁止读取外部文件 / 访问网络。
        self._conn.execute("SET enable_external_access=false")

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    def register_dataframe(self, name: str, frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame 必须是 pandas.DataFrame")
        self._conn.register(name, frame)

    def execute(self, sql: str, *, max_rows: int | None = None) -> pd.DataFrame:
        cleaned = ensure_read_only_sql(sql)
        # DEBT-001: 只读校验基于关键字/函数黑名单, 未做 SQL AST 解析。
        timer: threading.Timer | None = None
        if self._timeout_seconds > 0:
            timer = threading.Timer(self._timeout_seconds, self._conn.interrupt)
            timer.daemon = True
            timer.start()
        try:
            result = self._conn.execute(cleaned).df()
        except duckdb.InterruptException as exc:
            raise QueryTimeoutError(f"查询超过 {self._timeout_seconds:g} 秒被中断") from exc
        finally:
            if timer is not None:
                timer.cancel()

        if max_rows is not None and len(result) > max_rows:
            result = result.head(max_rows)
        return result

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> DuckDBEngine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        connection: Any = getattr(self, "_conn", None)
        if connection is not None:
            connection.close()
