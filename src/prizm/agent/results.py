"""单轮问答的对外结果类型（从 orchestrator 拆出，避免与 graph 循环依赖）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from prizm.agent.clarify import ClarificationRequest
from prizm.agent.repair import RepairAttempt
from prizm.agent.trace import Trace


@dataclass(frozen=True, slots=True)
class Answer:
    question: str
    sql: str
    dataframe: pd.DataFrame
    chart: dict[str, Any] | None = None
    chart_png: bytes | None = None
    conclusion: str = ""
    conclusion_source: str = "template"
    attempts: int = 1
    repair_history: tuple[RepairAttempt, ...] = ()
    trace: Trace | None = None

    @property
    def empty(self) -> bool:
        return bool(self.dataframe.empty)

    @property
    def repaired(self) -> bool:
        return self.attempts > 1


@dataclass(frozen=True, slots=True)
class Clarification:
    question: str
    request: ClarificationRequest
    remaining: int = 0
    trace: Trace | None = None


def format_result_table(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    """把结果 DataFrame 渲染成可直接展示的文本表格。"""
    if frame.empty:
        return "（查询结果为空）"
    table = cast(str, frame.head(max_rows).to_string(index=False))
    if len(frame) > max_rows:
        table += f"\n… 仅显示前 {max_rows} 行，共 {len(frame)} 行"
    return table
