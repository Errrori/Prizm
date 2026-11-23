"""执行失败后的自修复：把格式化错误回传模型，重新规划并执行。

重试口径：总尝试次数不超过 :data:`MAX_REPAIR_ATTEMPTS`（含首次），
即首次失败后最多再修复两次。LLM 调用失败、额度超限等不可恢复错误
不会进入重试循环，直接向上抛出。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import duckdb
import pandas as pd

from prizm.agent.planner import Plan, Planner, PlanParseError
from prizm.engine.duckdb_engine import DuckDBEngine, QueryTimeoutError, SQLValidationError

MAX_REPAIR_ATTEMPTS = 3
_MAX_ERROR_CHARS = 2_000
_MAX_SQL_CHARS = 4_000

_RETRYABLE_ERRORS = (PlanParseError, SQLValidationError, QueryTimeoutError, duckdb.Error)


class RepairFailed(RuntimeError):
    """达到最大尝试次数后仍无法执行。"""

    def __init__(
        self,
        *,
        question: str,
        attempts: int,
        last_error: str,
        history: tuple[RepairAttempt, ...] = (),
    ) -> None:
        self.question = question
        self.attempts = attempts
        self.last_error = last_error
        self.history = history
        super().__init__(f"共尝试 {attempts} 次后仍失败：{last_error}")


@dataclass(frozen=True, slots=True)
class RepairAttempt:
    """一次失败尝试的摘要，用于透明展示与测试断言。"""

    attempt: int
    sql: str | None
    error: str


@dataclass(frozen=True, slots=True)
class RepairResult:
    plan: Plan
    dataframe: pd.DataFrame
    attempts: int
    history: tuple[RepairAttempt, ...] = ()


def format_repair_error(exc: BaseException) -> str:
    """只保留错误类型和消息，不回传本机 traceback / 文件路径。"""
    text = f"{type(exc).__name__}: {exc}".strip()
    if len(text) > _MAX_ERROR_CHARS:
        text = text[:_MAX_ERROR_CHARS] + "…"
    return text


def build_repair_feedback(*, attempt: int, failed_sql: str | None, error: str) -> str:
    parts = [f"第 {attempt} 次尝试失败。", f"错误：{error}"]
    if failed_sql:
        sql = failed_sql
        if len(sql) > _MAX_SQL_CHARS:
            sql = sql[:_MAX_SQL_CHARS] + "…"
        parts.append(f"失败的 SQL：\n{sql}")
    parts.append("请修正上述问题，重新输出一个完整 JSON 计划，不要重复同样的错误。")
    return "\n".join(parts)


@dataclass
class Repairer:
    """围绕 Planner + DuckDBEngine 的确定性重试循环。"""

    planner: Planner
    engine: DuckDBEngine
    max_attempts: int = MAX_REPAIR_ATTEMPTS

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or not 1 <= self.max_attempts <= MAX_REPAIR_ATTEMPTS
        ):
            raise ValueError(f"max_attempts 必须在 1 到 {MAX_REPAIR_ATTEMPTS} 之间")

    def run(
        self,
        question: str,
        view_name: str,
        frame: pd.DataFrame,
        *,
        profile_summary: str | None = None,
        confirmed_policies: Mapping[str, str] | None = None,
        conversation_summary: str | None = None,
    ) -> RepairResult:
        history: list[RepairAttempt] = []
        feedback: str | None = None
        last_error = ""

        for attempt in range(1, self.max_attempts + 1):
            plan: Plan | None = None
            try:
                plan = self.planner.plan(
                    question,
                    view_name,
                    frame,
                    profile_summary=profile_summary,
                    confirmed_policies=confirmed_policies,
                    conversation_summary=conversation_summary,
                    repair_feedback=feedback,
                )
                dataframe = self.engine.execute(plan.sql)
                return RepairResult(
                    plan=plan,
                    dataframe=dataframe,
                    attempts=attempt,
                    history=tuple(history),
                )
            except _RETRYABLE_ERRORS as exc:
                last_error = format_repair_error(exc)
                failed_sql = plan.sql if plan is not None else None
                history.append(
                    RepairAttempt(attempt=attempt, sql=failed_sql, error=last_error)
                )
                if attempt >= self.max_attempts:
                    raise RepairFailed(
                        question=question,
                        attempts=attempt,
                        last_error=last_error,
                        history=tuple(history),
                    ) from exc
                feedback = build_repair_feedback(
                    attempt=attempt,
                    failed_sql=failed_sql,
                    error=last_error,
                )

        raise RepairFailed(  # pragma: no cover - 循环必然在内部返回或抛出
            question=question,
            attempts=self.max_attempts,
            last_error=last_error,
        )
