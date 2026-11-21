import pandas as pd
import pytest

from prizm.engine.duckdb_engine import (
    DuckDBEngine,
    QueryTimeoutError,
    SQLValidationError,
    ensure_read_only_sql,
)


def test_register_and_execute() -> None:
    with DuckDBEngine() as engine:
        engine.register_dataframe("t", pd.DataFrame({"a": [1, 2, 3]}))
        result = engine.execute("SELECT SUM(a) AS total FROM t")

    assert int(result.loc[0, "total"]) == 6


def test_execute_truncates_to_max_rows() -> None:
    with DuckDBEngine() as engine:
        engine.register_dataframe("t", pd.DataFrame({"a": range(10)}))
        result = engine.execute("SELECT a FROM t ORDER BY a", max_rows=3)

    assert result["a"].tolist() == [0, 1, 2]


def test_read_only_validation_strips_trailing_semicolon() -> None:
    assert ensure_read_only_sql("SELECT 1;") == "SELECT 1"


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "DELETE FROM t",
        "DROP TABLE t",
        "SELECT 1; SELECT 2",
        "WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x",
    ],
)
def test_read_only_validation_rejects_unsafe_sql(sql: str) -> None:
    with pytest.raises(SQLValidationError):
        ensure_read_only_sql(sql)


def test_read_only_validation_rejects_file_functions() -> None:
    with pytest.raises(SQLValidationError, match="read_csv"):
        ensure_read_only_sql("SELECT * FROM read_csv_auto('data.csv')")


@pytest.mark.parametrize(
    "sql",
    [
        "WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x",
        "WITH x AS (INSERT INTO t VALUES (1)) SELECT 1",
        "WITH x AS (UPDATE t SET a = 1) SELECT 1",
        "WITH x AS (COPY t TO 'out.csv') SELECT 1",
    ],
)
def test_read_only_validation_rejects_dml_inside_cte(sql: str) -> None:
    with pytest.raises(SQLValidationError, match="禁止的 SQL 关键字"):
        ensure_read_only_sql(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM query('SELECT 1')",
        "SELECT * FROM query_table('t')",
        "SELECT * FROM json_execute_serialized_sql('x')",
        "SELECT * FROM read_text('x')",
        "SELECT * FROM read_blob('x')",
        "SELECT * FROM sniff_csv('x.csv')",
    ],
)
def test_read_only_validation_rejects_dynamic_and_file_functions(sql: str) -> None:
    with pytest.raises(SQLValidationError, match="禁止的 SQL 函数"):
        ensure_read_only_sql(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 'drop table t' AS message",
        "SELECT 1 -- delete from t",
        "SELECT 1 /* update t */",
        'SELECT "delete" FROM t',
        "SELECT 1 AS update",
        "SELECT ';'",
    ],
)
def test_read_only_validation_ignores_keywords_in_literals_and_comments(sql: str) -> None:
    assert ensure_read_only_sql(sql) == sql


def test_read_only_validation_strips_trailing_semicolons_in_literal_sql() -> None:
    assert ensure_read_only_sql("SELECT ';';") == "SELECT ';'"


def test_read_only_validation_reports_parse_errors() -> None:
    with pytest.raises(SQLValidationError, match="SQL 解析失败"):
        ensure_read_only_sql("SELECT FROM")


def test_invalid_limits_are_rejected() -> None:
    with pytest.raises(ValueError, match="memory_limit"):
        DuckDBEngine(memory_limit="a lot")
    with pytest.raises(ValueError, match="threads"):
        DuckDBEngine(threads=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        DuckDBEngine(timeout_seconds=-1)


def test_query_timeout_interrupts_long_query() -> None:
    with DuckDBEngine(timeout_seconds=0.1) as engine, pytest.raises(QueryTimeoutError):
        engine.execute(
            "SELECT SUM(a.i * b.i) "
            "FROM range(1000000000) a(i), range(1000000000) b(i)"
        )
