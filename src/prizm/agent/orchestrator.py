"""最薄端到端编排：连接器 → 体检 → 澄清门禁 → Planner → 执行 → 结果。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from prizm.agent.clarify import ClarificationOption, ClarificationRequest, ClarifyGate
from prizm.agent.memory import ConversationMemory, Turn
from prizm.agent.planner import Planner
from prizm.agent.repair import RepairAttempt, Repairer
from prizm.datasource.base import BaseConnector
from prizm.engine.duckdb_engine import DuckDBEngine
from prizm.profiling.models import ProfileReport
from prizm.profiling.profiler import ProfileOptions, profile_frame
from prizm.rag.retriever import DictionaryAnswer, DictionaryQA, is_dictionary_question

_HISTORY_PREVIEW_ROWS = 3


@dataclass(frozen=True, slots=True)
class Answer:
    question: str
    sql: str
    dataframe: pd.DataFrame
    chart: dict[str, Any] | None = None
    attempts: int = 1
    repair_history: tuple[RepairAttempt, ...] = ()

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


@dataclass
class _PendingClarification:
    question: str
    requests: tuple[ClarificationRequest, ...]
    index: int = 0

    @property
    def current(self) -> ClarificationRequest:
        return self.requests[self.index]

    @property
    def remaining_after(self) -> int:
        return len(self.requests) - self.index - 1

    @property
    def done(self) -> bool:
        return self.index >= len(self.requests)

    def advance(self) -> None:
        self.index += 1


class Orchestrator:
    """单数据集、多轮问答；上传后立即体检，提问时先过澄清门禁。"""

    def __init__(
        self,
        connector: BaseConnector,
        engine: DuckDBEngine,
        planner: Planner,
        *,
        profile_options: ProfileOptions | None = None,
        primary_key: str | Sequence[str] | None = None,
        clarify_gate: ClarifyGate | None = None,
        repairer: Repairer | None = None,
        memory: ConversationMemory | None = None,
        dictionary_qa: DictionaryQA | None = None,
    ) -> None:
        self._frame = connector.load()
        self._engine = engine
        self._planner = planner
        self._repairer = repairer or Repairer(planner, engine)
        self._memory = memory or ConversationMemory()
        self._dictionary_qa = dictionary_qa
        self._profile_options = profile_options or ProfileOptions()
        self._profile = profile_frame(
            self._frame,
            primary_key=primary_key,
            options=self._profile_options,
        )
        connector.register(engine)
        self.view_name = connector.view_name
        self._clarify_gate = clarify_gate or ClarifyGate()
        self._pending: _PendingClarification | None = None

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._frame

    @property
    def profile(self) -> ProfileReport:
        return self._profile

    @property
    def history(self) -> tuple[Turn, ...]:
        return self._memory.turns

    def attach_dictionary(self, dictionary_qa: DictionaryQA | None) -> None:
        """挂载/替换数据字典问答；None 表示关闭 RAG 路由。"""
        self._dictionary_qa = dictionary_qa

    @property
    def pending_clarification(self) -> Clarification | None:
        if self._pending is None:
            return None
        return self._as_clarification(self._pending)

    def ask(self, question: str) -> Answer | Clarification | DictionaryAnswer:
        if not question.strip():
            raise ValueError("问题不能为空")
        if self._pending is not None:
            raise RuntimeError("存在未完成的澄清，请先回答澄清问题")
        text = question.strip()
        if self._dictionary_qa is not None and is_dictionary_question(text):
            return self._dictionary_qa.answer(text)
        return self._advance(text)

    def resolve(self, value: str) -> Answer | Clarification:
        """记录用户对当前澄清项的回复；全部澄清完成后重跑原问题。"""
        pending = self._pending
        if pending is None:
            raise RuntimeError("当前没有待澄清的问题")
        option = _match_option(pending.current, value)
        self._clarify_gate.confirm(
            pending.current.code,
            option.label,
            question=pending.question,
        )
        pending.advance()
        if not pending.done:
            return self._as_clarification(pending)
        self._pending = None
        return self._advance(pending.question)

    def _advance(self, question: str) -> Answer | Clarification:
        decision = self._clarify_gate.judge(question, self._profile)
        if decision.needs_clarification:
            self._pending = _PendingClarification(question, decision.requests)
            return self._as_clarification(self._pending)
        return self._execute(question)

    def _execute(self, question: str) -> Answer:
        policies = self._clarify_gate.policy_summary()
        outcome = self._repairer.run(
            question,
            self.view_name,
            self._frame,
            profile_summary=self._profile.to_prompt(
                max_chars=self._profile_options.max_prompt_chars
            ),
            confirmed_policies=policies or None,
            conversation_summary=self._memory.format_for_prompt(),
        )
        answer = Answer(
            question=question,
            sql=outcome.plan.sql,
            dataframe=outcome.dataframe,
            chart=outcome.plan.chart,
            attempts=outcome.attempts,
            repair_history=outcome.history,
        )
        self._memory.add(
            Turn(
                question=question,
                sql=answer.sql,
                columns=tuple(str(column) for column in answer.dataframe.columns),
                row_count=len(answer.dataframe),
                preview=tuple(
                    {str(key): value for key, value in row.items()}
                    for row in answer.dataframe.head(_HISTORY_PREVIEW_ROWS).to_dict(
                        orient="records"
                    )
                ),
                chart=answer.chart,
            )
        )
        return answer

    @staticmethod
    def _as_clarification(pending: _PendingClarification) -> Clarification:
        return Clarification(
            question=pending.question,
            request=pending.current,
            remaining=pending.remaining_after,
        )


def _match_option(request: ClarificationRequest, value: str) -> ClarificationOption:
    text = value.strip()
    for option in request.options:
        if text in {option.value, option.label}:
            return option
    raise ValueError(f"无效的澄清选项: {value!r}")


def format_result_table(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    """把结果 DataFrame 渲染成可直接展示的文本表格。"""
    if frame.empty:
        return "（查询结果为空）"
    table = cast(str, frame.head(max_rows).to_string(index=False))
    if len(frame) > max_rows:
        table += f"\n… 仅显示前 {max_rows} 行，共 {len(frame)} 行"
    return table
