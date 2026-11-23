import json
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd
import pytest

from prizm.agent.planner import Planner
from prizm.agent.repair import (
    MAX_REPAIR_ATTEMPTS,
    Repairer,
    RepairFailed,
    build_repair_feedback,
    format_repair_error,
)
from prizm.engine.duckdb_engine import (
    DuckDBEngine,
    QueryTimeoutError,
    SQLValidationError,
)
from prizm.llm.adapter import LLMError, Message, MockLLM
from prizm.utils.tokens import Completion, QuotaExceededError


@dataclass
class ScriptedLLM:
    responses: list[str]
    calls: list[list[Message]] = field(default_factory=list)

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> Completion:
        self.calls.append([dict(message) for message in messages])
        if not self.responses:
            raise AssertionError("没有更多脚本化响应")
        return Completion(self.responses.pop(0))


@dataclass
class FailingLLM:
    error: BaseException
    calls: int = 0

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> Completion:
        self.calls += 1
        raise self.error


@dataclass
class TimeoutOnceEngine:
    calls: int = 0

    def execute(self, sql: str) -> pd.DataFrame:
        self.calls += 1
        if self.calls == 1:
            raise QueryTimeoutError("查询超过 0.1 秒被中断")
        return pd.DataFrame({"ok": [self.calls]})


def frame() -> pd.DataFrame:
    return pd.DataFrame({"status": ["已完成", "已取消"], "sales": [100, 200]})


def plan(sql: str) -> str:
    return json.dumps({"sql": sql, "chart": None})


def test_repairer_returns_first_attempt_without_retry() -> None:
    llm = ScriptedLLM([plan("SELECT SUM(sales) AS total FROM orders")])
    data = frame()

    with DuckDBEngine() as engine:
        engine.register_dataframe("orders", data)
        result = Repairer(Planner(llm), engine).run("销售额", "orders", data)

    assert result.attempts == 1
    assert result.history == ()
    assert int(result.dataframe.loc[0, "total"]) == 300
    assert len(llm.calls) == 1


def test_repairer_retries_after_sql_execution_error() -> None:
    bad_sql = "SELECT missing_column FROM orders"
    good_sql = "SELECT SUM(sales) AS total FROM orders"
    llm = ScriptedLLM([plan(bad_sql), plan(good_sql)])
    data = frame()

    with DuckDBEngine() as engine:
        engine.register_dataframe("orders", data)
        result = Repairer(Planner(llm), engine).run("销售额", "orders", data)

    assert result.attempts == 2
    assert len(result.history) == 1
    assert result.history[0].attempt == 1
    assert result.history[0].sql == bad_sql
    assert "missing_column" in result.history[0].error
    assert int(result.dataframe.loc[0, "total"]) == 300

    repair_prompt = llm.calls[1][1]["content"]
    assert "上一次尝试失败" in repair_prompt
    assert bad_sql in repair_prompt
    assert "missing_column" in repair_prompt


def test_repairer_retries_after_sql_validation_error() -> None:
    llm = ScriptedLLM([plan("DELETE FROM orders"), plan("SELECT 1 AS ok")])
    data = frame()

    with DuckDBEngine() as engine:
        engine.register_dataframe("orders", data)
        result = Repairer(Planner(llm), engine).run("销售额", "orders", data)

    assert result.attempts == 2
    assert result.history[0].sql == "DELETE FROM orders"
    assert "只允许" in result.history[0].error
    assert int(result.dataframe.loc[0, "ok"]) == 1


def test_repairer_retries_after_query_timeout() -> None:
    engine = TimeoutOnceEngine()
    llm = ScriptedLLM([plan("SELECT 1 AS ok"), plan("SELECT 2 AS ok")])

    result = Repairer(Planner(llm), engine).run("随便看看", "orders", pd.DataFrame({"a": [1]}))

    assert engine.calls == 2
    assert result.attempts == 2
    assert "QueryTimeoutError" in result.history[0].error
    assert int(result.dataframe.loc[0, "ok"]) == 2


def test_repairer_keeps_conversation_summary_on_every_attempt() -> None:
    llm = ScriptedLLM(
        [
            plan("SELECT missing_column FROM orders"),
            plan("SELECT 1 AS ok"),
        ]
    )
    data = frame()

    with DuckDBEngine() as engine:
        engine.register_dataframe("orders", data)
        Repairer(Planner(llm), engine).run(
            "再算一次",
            "orders",
            data,
            conversation_summary="上一轮 SQL：SELECT 1",
        )

    assert "上一轮 SQL：SELECT 1" in llm.calls[0][1]["content"]
    assert "上一轮 SQL：SELECT 1" in llm.calls[1][1]["content"]


def test_repairer_retries_after_plan_parse_error() -> None:
    llm = ScriptedLLM(["不是 JSON", plan("SELECT 1 AS ok")])
    data = frame()

    with DuckDBEngine() as engine:
        engine.register_dataframe("orders", data)
        result = Repairer(Planner(llm), engine).run("随便看看", "orders", data)

    assert result.attempts == 2
    assert result.history[0].sql is None
    assert "PlanParseError" in result.history[0].error
    assert "上一次尝试失败" in llm.calls[1][1]["content"]


def test_repairer_exhausts_max_attempts() -> None:
    llm = ScriptedLLM([plan("SELECT missing FROM orders")] * MAX_REPAIR_ATTEMPTS)
    data = frame()

    with DuckDBEngine() as engine:
        engine.register_dataframe("orders", data)
        with pytest.raises(RepairFailed) as excinfo:
            Repairer(Planner(llm), engine).run("销售额", "orders", data)

    assert excinfo.value.attempts == MAX_REPAIR_ATTEMPTS
    assert "missing" in excinfo.value.last_error
    assert len(llm.calls) == MAX_REPAIR_ATTEMPTS


def test_repairer_does_not_retry_llm_failure() -> None:
    llm = FailingLLM(LLMError("模型抖动"))
    data = frame()

    with DuckDBEngine() as engine:
        engine.register_dataframe("orders", data)
        with pytest.raises(LLMError, match="模型抖动"):
            Repairer(Planner(llm), engine).run("销售额", "orders", data)

    assert llm.calls == 1


def test_repairer_does_not_retry_quota_failure() -> None:
    llm = FailingLLM(QuotaExceededError("额度用完"))
    data = frame()

    with DuckDBEngine() as engine:
        engine.register_dataframe("orders", data)
        with pytest.raises(QuotaExceededError, match="额度用完"):
            Repairer(Planner(llm), engine).run("销售额", "orders", data)

    assert llm.calls == 1


@pytest.mark.parametrize("attempts", [0, MAX_REPAIR_ATTEMPTS + 1])
def test_repairer_rejects_invalid_max_attempts(attempts: int) -> None:
    with DuckDBEngine() as engine, pytest.raises(ValueError, match="max_attempts"):
        Repairer(Planner(MockLLM()), engine, max_attempts=attempts)


def test_format_repair_error_uses_type_and_message_only() -> None:
    text = format_repair_error(SQLValidationError("只允许 SELECT"))

    assert text == "SQLValidationError: 只允许 SELECT"
    assert "Traceback" not in text
    assert "C:\\" not in text


def test_build_repair_feedback_includes_error_and_sql() -> None:
    feedback = build_repair_feedback(
        attempt=1,
        failed_sql="SELECT bad FROM orders",
        error="BinderException: bad",
    )

    assert "第 1 次尝试失败" in feedback
    assert "BinderException: bad" in feedback
    assert "SELECT bad FROM orders" in feedback
