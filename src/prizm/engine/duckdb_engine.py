"""DuckDB 只读执行引擎：注册 DataFrame 视图 + 限额执行 SQL。"""

from __future__ import annotations

import re
import threading
from types import TracebackType
from typing import Any

import duckdb
import pandas as pd

_MEMORY_LIMIT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:B|[KMGT]i?B)", re.IGNORECASE)
_FIRST_TOKEN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_$]*)")

# DuckDB 允许 WITH CTE 中出现 INSERT/UPDATE/DELETE/COPY, 顶层语句类型仍是 SELECT,
# 因此除语句类型白名单外, 还要在词法层拦截可能写入的关键字。
_FORBIDDEN_KEYWORDS = frozenset(
    {
        "alter",
        "analyze",
        "attach",
        "begin",
        "call",
        "checkpoint",
        "commit",
        "copy",
        "create",
        "delete",
        "detach",
        "drop",
        "execute",
        "export",
        "grant",
        "import",
        "insert",
        "install",
        "load",
        "merge",
        "pragma",
        "prepare",
        "revoke",
        "rollback",
        "set",
        "truncate",
        "update",
        "vacuum",
    }
)

# 即使 enable_external_access=false, 也显式拦截可能触碰文件/网络/动态 SQL 的表函数。
# DEBT-012: 危险函数仍按名单拦截, 升级 DuckDB 或发现新绕过时需同步。
_FORBIDDEN_FUNCTIONS = frozenset(
    {
        "arrow_scan",
        "arrow_scan_dumb",
        "checkpoint",
        "create_external_resource",
        "delta_scan",
        "destroy_external_resource",
        "disable_logging",
        "disable_profiling",
        "enable_logging",
        "enable_profiling",
        "force_checkpoint",
        "glob",
        "httpfs",
        "iceberg_scan",
        "json_execute_serialized_sql",
        "mysql_scan",
        "parquet_bloom_probe",
        "parquet_file_metadata",
        "parquet_full_metadata",
        "parquet_kv_metadata",
        "parquet_metadata",
        "parquet_scan",
        "parquet_schema",
        "postgres_scan",
        "pandas_scan",
        "python_arrow_scan",
        "python_map_function",
        "query",
        "query_table",
        "read_blob",
        "read_csv",
        "read_csv_auto",
        "read_duckdb",
        "read_json",
        "read_json_auto",
        "read_ndjson",
        "read_ndjson_auto",
        "read_parquet",
        "read_single_csv_file",
        "read_single_json_file",
        "read_single_parquet_file",
        "read_text",
        "register_external_resource",
        "register_external_resource_type",
        "sniff_csv",
        "sqlite_scan",
        "truncate_duckdb_logs",
    }
)

_FORBIDDEN_KEYWORD_RE = re.compile(
    r"(?:^|\()\s*(?P<keyword>" + "|".join(sorted(map(re.escape, _FORBIDDEN_KEYWORDS))) + r")\b",
    re.IGNORECASE,
)
_FORBIDDEN_FUNCTION_RE = re.compile(
    r"\b(?P<function>" + "|".join(sorted(map(re.escape, _FORBIDDEN_FUNCTIONS))) + r")\s*\(",
    re.IGNORECASE,
)


class SQLValidationError(ValueError):
    """SQL 未通过只读校验。"""


class QueryTimeoutError(RuntimeError):
    """查询超过配置的超时时间。"""


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _strip_noncode(sql: str) -> str:
    """把注释、字符串和引号标识符替换为占位符, 只保留 SQL 词法骨架。

    这样关键字/函数扫描不会被字符串或注释里的内容误伤, 也不会漏掉
    ``WITH x AS (DELETE ...)`` 这类藏在 CTE 里的写操作。
    """
    out: list[str] = []
    index = 0
    length = len(sql)
    while index < length:
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < length else ""

        if char == "-" and next_char == "-":
            index += 2
            while index < length and sql[index] not in "\r\n":
                index += 1
            out.append(" ")
            continue

        if char == "/" and next_char == "*":
            depth = 1
            index += 2
            while index < length and depth > 0:
                if sql.startswith("/*", index):
                    depth += 1
                    index += 2
                elif sql.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            out.append(" ")
            continue

        if char in {"'", '"', "`"}:
            quote = char
            index += 1
            while index < length:
                if sql[index] == quote:
                    if index + 1 < length and sql[index + 1] == quote:
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            out.append("''" if quote == "'" else '""')
            continue

        if char == "$":
            tag_end = index + 1
            while tag_end < length and (sql[tag_end].isalnum() or sql[tag_end] == "_"):
                tag_end += 1
            if tag_end < length and sql[tag_end] == "$":
                tag = sql[index : tag_end + 1]
                close = sql.find(tag, tag_end + 1)
                if close != -1:
                    index = close + len(tag)
                    out.append("''")
                    continue

        out.append(char)
        index += 1
    return "".join(out)


def _strip_trailing_semicolons(sql: str) -> str:
    cleaned = sql.strip()
    while cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def ensure_read_only_sql(sql: str) -> str:
    """校验只允许单条 SELECT / WITH 查询，返回去掉尾部分号的 SQL。

    DEBT-001: 用 DuckDB parser 做语句类型白名单, 再对去字符串/注释后的
    词法骨架拦截 CTE 内的写操作和危险表函数; 仍非完整 AST 解析(见 DEBT-012)。
    """
    cleaned = sql.strip()
    if not cleaned:
        raise SQLValidationError("SQL 不能为空")

    try:
        statements = duckdb.extract_statements(cleaned)
    except duckdb.Error as exc:
        raise SQLValidationError(f"SQL 解析失败: {exc}") from exc
    if len(statements) != 1:
        raise SQLValidationError("只允许执行单条 SQL 语句")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise SQLValidationError(
            f"只允许 SELECT / WITH 查询，当前语句类型: {statements[0].type}"
        )

    stripped = _strip_noncode(cleaned)
    first_token = _FIRST_TOKEN_RE.match(stripped)
    if first_token is None or first_token.group(1).lower() not in {"select", "with"}:
        raise SQLValidationError("只允许 SELECT / WITH 查询")

    keyword = _FORBIDDEN_KEYWORD_RE.search(stripped)
    if keyword is not None:
        raise SQLValidationError(f"禁止的 SQL 关键字: {keyword.group('keyword').upper()}")
    function = _FORBIDDEN_FUNCTION_RE.search(stripped)
    if function is not None:
        raise SQLValidationError(f"禁止的 SQL 函数: {function.group('function')}")
    return _strip_trailing_semicolons(cleaned)


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
        # DEBT-012: 只读校验仍非完整 AST, 危险函数按名单拦截。
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
