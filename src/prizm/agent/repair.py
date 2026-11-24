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
from prizm.agent.progress import ProgressEvent, ProgressSink
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


def _emit_start(progress: ProgressSink | None, name: str, summary: str) -> None:
    if progress is not None:
        progress(ProgressEvent(kind="step_start", name=name, summary=summary))


def _emit_end(
    progress: ProgressSink | None,
    name: str,
    status: str,
    summary: str,
    detail: str | None = None,
) -> None:
    if progress is not None:
        progress(
            ProgressEvent(
                kind="step",
                name=name,
                status=status,
                summary=summary,
                detail=detail,
            )
        )


def _hint_for_error(error: str) -> str | None:
    lowered = error.lower()
    if "date_part" in lowered or (
        "no function matches" in lowered and "varchar" in lowered
    ):
        return (
            "提示：月份/日期列可能是 VARCHAR，请用 substr(列, 1, 7) AS month，"
            "或先 CAST(列 AS TIMESTAMP)。"
        )
    if "from clause is missing" in lowered or "not found in from clause" in lowered:
        return "提示：聚合和列引用必须带 FROM 表名；CTE 外层只能引用 CTE 已选出的列。"
    if "must appear in the group by clause" in lowered:
        return "提示：SELECT 中的非聚合列必须出现在 GROUP BY，或改用聚合表达式。"
    if "syntax error" in lowered or "parser error" in lowered:
        return "提示：WITH ... AS (...) 之后必须还有最终 SELECT，并检查括号是否配对。"
    return None


def build_repair_feedback(*, attempt: int, failed_sql: str | None, error: str) -> str:
    parts = [f"第 {attempt} 次尝试失败。", f"错误：{error}"]
    if failed_sql:
        sql = failed_sql
        if len(sql) > _MAX_SQL_CHARS:
            sql = sql[:_MAX_SQL_CHARS] + "…"
        parts.append(f"失败的 SQL：\n{sql}")
    hint = _hint_for_error(error)
    if hint:
        parts.append(hint)
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
        progress: ProgressSink | None = None,
    ) -> RepairResult:
        history: list[RepairAttempt] = []
        feedback: str | None = None
        last_error = ""

        for attempt in range(1, self.max_attempts + 1):
            plan: Plan | None = None
            _emit_start(
                progress,
                "SQL 规划",
                "正在生成 SQL…" if attempt == 1 else f"第 {attempt} 次重新生成 SQL…",
            )
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
                _emit_end(progress, "SQL 规划", "ok", "已生成只读 SQL", plan.sql)
                _emit_start(progress, "执行", "正在执行 SQL…")
                dataframe = self.engine.execute(plan.sql)
                _emit_end(
                    progress,
                    "执行",
                    "ok",
                    f"返回 {len(dataframe)} 行 × {len(dataframe.columns)} 列",
                )
                return RepairResult(
                    plan=plan,
                    dataframe=dataframe,
                    attempts=attempt,
                    history=tuple(history),
                )
            except _RETRYABLE_ERRORS as exc:
                last_error = format_repair_error(exc)
                failed_sql = plan.sql if plan is not None else None
                if plan is None:
                    _emit_end(progress, "SQL 规划", "failed", "SQL 生成失败", last_error)
                else:
                    _emit_end(progress, "执行", "failed", "SQL 执行失败", last_error)
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
                _emit_end(
                    progress,
                    "自修复",
                    "retrying",
                    f"第 {attempt} 次失败，准备重试",
                    last_error,
                )
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
