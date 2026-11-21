"""最薄端到端编排：连接器 → 体检 → DuckDB → Planner → 执行 → 结果。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from prizm.agent.planner import Planner
from prizm.datasource.base import BaseConnector
from prizm.engine.duckdb_engine import DuckDBEngine
from prizm.profiling.models import ProfileReport
from prizm.profiling.profiler import ProfileOptions, profile_frame


@dataclass(frozen=True)
class Answer:
    question: str
    sql: str
    dataframe: pd.DataFrame
    chart: dict[str, Any] | None = None

    @property
    def empty(self) -> bool:
        return bool(self.dataframe.empty)


class Orchestrator:
    """单数据集、单轮问答；上传后立即体检，提问时注入紧凑摘要。"""

    def __init__(
        self,
        connector: BaseConnector,
        engine: DuckDBEngine,
        planner: Planner,
        *,
        profile_options: ProfileOptions | None = None,
        primary_key: str | Sequence[str] | None = None,
    ) -> None:
        self._frame = connector.load()
        self._engine = engine
        self._planner = planner
        self._profile_options = profile_options or ProfileOptions()
        self._profile = profile_frame(
            self._frame,
            primary_key=primary_key,
            options=self._profile_options,
        )
        connector.register(engine)
        self.view_name = connector.view_name

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._frame

    @property
    def profile(self) -> ProfileReport:
        return self._profile

    def ask(self, question: str) -> Answer:
        if not question.strip():
            raise ValueError("问题不能为空")
        plan = self._planner.plan(
            question.strip(),
            self.view_name,
            self._frame,
            profile_summary=self._profile.to_prompt(
                max_chars=self._profile_options.max_prompt_chars
            ),
        )
        result = self._engine.execute(plan.sql)
        return Answer(
            question=question.strip(),
            sql=plan.sql,
            dataframe=result,
            chart=plan.chart,
        )


def format_result_table(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    """把结果 DataFrame 渲染成可直接展示的文本表格。"""
    if frame.empty:
        return "（查询结果为空）"
    table = cast(str, frame.head(max_rows).to_string(index=False))
    if len(frame) > max_rows:
        table += f"\n… 仅显示前 {max_rows} 行，共 {len(frame)} 行"
    return table
