"""最薄端到端编排：连接器 → 体检 → 澄清门禁 → Planner → 执行 → 结果。"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from prizm.agent.clarify import ClarificationOption, ClarificationRequest, ClarifyGate
from prizm.agent.composer import Conclusion, ConclusionComposer, template_conclusion
from prizm.agent.memory import ConversationMemory, Turn
from prizm.agent.planner import Planner
from prizm.agent.progress import ProgressEvent, ProgressSink, TokenSink
from prizm.agent.repair import RepairAttempt, Repairer, RepairFailed
from prizm.agent.trace import (
    ProfileReference,
    Trace,
    TraceStep,
    policy_label,
)
from prizm.datasource.base import BaseConnector
from prizm.engine.duckdb_engine import DuckDBEngine
from prizm.profiling.models import Anomaly, ProfileReport
from prizm.profiling.profiler import ProfileOptions, profile_frame
from prizm.rag.retriever import DictionaryAnswer, DictionaryQA, is_dictionary_question

if TYPE_CHECKING:
    from prizm.viz.charts import ChartSpec

_HISTORY_PREVIEW_ROWS = 3


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
        conclusion_composer: ConclusionComposer | None = None,
    ) -> None:
        self._frame = connector.load()
        self._engine = engine
        self._planner = planner
        self._repairer = repairer or Repairer(planner, engine)
        self._memory = memory or ConversationMemory()
        self._dictionary_qa = dictionary_qa
        self._conclusion_composer = conclusion_composer
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
        self._last_trace: Trace | None = None

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._frame

    @property
    def profile(self) -> ProfileReport:
        return self._profile

    @property
    def history(self) -> tuple[Turn, ...]:
        return self._memory.turns

    @property
    def last_trace(self) -> Trace | None:
        """最近一次 ask/resolve 的透明 trace，供 UI 展示。"""
        return self._last_trace

    def attach_dictionary(self, dictionary_qa: DictionaryQA | None) -> None:
        """挂载/替换数据字典问答；None 表示关闭 RAG 路由。"""
        self._dictionary_qa = dictionary_qa

    @property
    def pending_clarification(self) -> Clarification | None:
        if self._pending is None:
            return None
        return self._as_clarification(self._pending, self._last_trace)

    def ask(
        self,
        question: str,
        *,
        progress: ProgressSink | None = None,
    ) -> Answer | Clarification | DictionaryAnswer:
        if not question.strip():
            raise ValueError("问题不能为空")
        if self._pending is not None:
            raise RuntimeError("存在未完成的澄清，请先回答澄清问题")
        text = question.strip()
        if self._dictionary_qa is not None and is_dictionary_question(text):
            answer = self._dictionary_qa.answer(text)
            steps = (
                TraceStep("路由判定", "ok", "命中数据字典术语问题，走检索而非 SQL"),
                TraceStep(
                    "字典检索",
                    "ok",
                    f"返回 {len(answer.citations)} 条可溯源片段",
                    detail=answer.text,
                ),
            )
            self._last_trace = Trace(question=text, steps=steps)
            for step in steps:
                _emit_step(progress, step)
            return answer
        return self._advance(text, progress=progress)

    def resolve(
        self,
        value: str,
        *,
        progress: ProgressSink | None = None,
    ) -> Answer | Clarification:
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
        policy_name = policy_label(pending.current.code)
        step = TraceStep(
            "按确认口径处理",
            "ok",
            f"{policy_name}：{option.label}",
            detail=option.description or None,
        )
        prior_steps: list[TraceStep] = []
        prior_references: tuple[ProfileReference, ...] = ()
        if self._last_trace is not None and self._last_trace.question == pending.question:
            prior_steps = list(self._last_trace.steps)
            prior_references = self._last_trace.profile_references
        steps = list(prior_steps)
        _append_step(steps, step, progress)
        pending.advance()
        references = _merge_references(
            prior_references,
            _profile_references(self._profile, pending.requests),
        )
        if not pending.done:
            trace = self._clarification_trace(pending, steps, references)
            self._last_trace = trace
            return self._as_clarification(pending, trace)
        self._pending = None
        return self._advance(pending.question, steps, references, progress=progress)

    def _advance(
        self,
        question: str,
        prior_steps: Sequence[TraceStep] = (),
        prior_references: Sequence[ProfileReference] = (),
        *,
        progress: ProgressSink | None = None,
    ) -> Answer | Clarification:
        decision = self._clarify_gate.judge(question, self._profile)
        steps = list(prior_steps)
        references = _merge_references(
            tuple(prior_references),
            _profile_references(self._profile, decision.requests),
        )
        if decision.needs_clarification:
            reasons = "；".join(request.reason for request in decision.requests)
            _append_step(
                steps,
                TraceStep(
                    "体检与澄清判定",
                    "clarify",
                    f"触发 {len(decision.requests)} 项澄清，暂不执行",
                    detail=reasons,
                ),
                progress,
            )
            self._pending = _PendingClarification(question, decision.requests)
            trace = self._clarification_trace(self._pending, steps, references)
            self._last_trace = trace
            return self._as_clarification(self._pending, trace)
        _append_step(
            steps,
            TraceStep(
                "体检与澄清判定",
                "ok",
                f"体检 {len(self._profile.anomalies)} 条异常，无新增待澄清项，进入规划",
            ),
            progress,
        )
        return self._execute(question, steps, references, progress=progress)

    def _execute(
        self,
        question: str,
        prior_steps: Sequence[TraceStep] = (),
        references: Sequence[ProfileReference] = (),
        *,
        progress: ProgressSink | None = None,
    ) -> Answer:
        policies = self._clarify_gate.policy_summary()
        steps = list(prior_steps)
        start = time.perf_counter()
        try:
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
        except RepairFailed as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            failed_sql = exc.history[-1].sql if exc.history else None
            _append_step(
                steps,
                TraceStep(
                    "SQL 规划与执行",
                    "failed",
                    f"已尝试 {exc.attempts} 次仍未成功",
                    detail=_format_repair_history(exc.history) or exc.last_error,
                ),
                progress,
            )
            self._last_trace = Trace(
                question=question,
                steps=tuple(steps),
                generated_sql=failed_sql,
                execution_status="failed",
                profile_references=tuple(references),
                confirmed_policies=tuple(policies.items()),
                attempts=exc.attempts,
                elapsed_ms=elapsed_ms,
            )
            raise

        _append_step(
            steps,
            TraceStep("SQL 规划", "ok", "已生成只读 SQL", detail=outcome.plan.sql),
            progress,
        )
        if outcome.attempts > 1:
            _append_step(
                steps,
                TraceStep(
                    "自修复",
                    "repaired",
                    f"第 {outcome.attempts} 次尝试成功",
                    detail=_format_repair_history(outcome.history) or None,
                ),
                progress,
            )
        dataframe = outcome.dataframe
        _append_step(
            steps,
            TraceStep(
                "执行",
                "ok",
                f"返回 {len(dataframe)} 行 × {len(dataframe.columns)} 列",
            ),
            progress,
        )

        chart_spec, chart_payload, chart_png, chart_step = _render_plan_chart(
            outcome.plan.chart, dataframe
        )
        if chart_step is not None:
            _append_step(steps, chart_step, progress)

        if self._conclusion_composer is None:
            conclusion = Conclusion(template_conclusion(dataframe))
            _append_step(
                steps,
                TraceStep("结论生成", "ok", "使用模板结论（未启用模型归纳）"),
                progress,
            )
        else:
            token_callback: TokenSink | None = None
            if progress is not None:

                def _emit_token(chunk: str) -> None:
                    progress(ProgressEvent(kind="conclusion_token", text=chunk))

                token_callback = _emit_token

            conclusion = self._conclusion_composer.compose(
                question,
                outcome.plan.sql,
                dataframe,
                chart=chart_spec,
                on_token=token_callback,
            )
            if conclusion.source == "llm":
                _append_step(
                    steps,
                    TraceStep(
                        "结论生成",
                        "ok",
                        "已用模型归纳结果",
                        detail=conclusion.text,
                    ),
                    progress,
                )
            else:
                _append_step(
                    steps,
                    TraceStep(
                        "结论生成",
                        "fallback",
                        "模型结论不可用，已回退模板",
                        detail=conclusion.fallback_reason,
                    ),
                    progress,
                )

        elapsed_ms = (time.perf_counter() - start) * 1000
        trace = Trace(
            question=question,
            steps=tuple(steps),
            generated_sql=outcome.plan.sql,
            execution_status="repaired" if outcome.attempts > 1 else "succeeded",
            profile_references=tuple(references),
            confirmed_policies=tuple(policies.items()),
            attempts=outcome.attempts,
            elapsed_ms=elapsed_ms,
        )
        self._last_trace = trace

        answer = Answer(
            question=question,
            sql=outcome.plan.sql,
            dataframe=dataframe,
            chart=chart_payload,
            chart_png=chart_png,
            conclusion=conclusion.text,
            conclusion_source=conclusion.source,
            attempts=outcome.attempts,
            repair_history=outcome.history,
            trace=trace,
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

    def _clarification_trace(
        self,
        pending: _PendingClarification,
        steps: Sequence[TraceStep],
        references: Sequence[ProfileReference] = (),
    ) -> Trace:
        return Trace(
            question=pending.question,
            steps=tuple(steps),
            execution_status="not_run",
            profile_references=tuple(references),
            confirmed_policies=tuple(self._clarify_gate.policy_summary().items()),
        )

    @staticmethod
    def _as_clarification(
        pending: _PendingClarification,
        trace: Trace | None = None,
    ) -> Clarification:
        return Clarification(
            question=pending.question,
            request=pending.current,
            remaining=pending.remaining_after,
            trace=trace,
        )


def _render_plan_chart(
    raw: dict[str, Any] | None,
    dataframe: pd.DataFrame,
) -> tuple[ChartSpec | None, dict[str, Any] | None, bytes | None, TraceStep | None]:
    """校验并渲染图表；图表属于展示层，失败不阻断分析。"""
    if raw is None:
        return None, None, None, None

    from prizm.viz.charts import (
        MAX_CHART_POINTS,
        ChartError,
        parse_chart_spec,
        render_chart,
    )

    try:
        spec = parse_chart_spec(raw, dataframe.columns)
        if spec is None:
            return None, None, None, None
        png = render_chart(spec, dataframe)
    except ChartError as exc:
        return (
            None,
            None,
            None,
            TraceStep(
                "图表",
                "skipped",
                "图表参数不可用，已改用表格展示",
                detail=str(exc),
            ),
        )

    detail = spec.title
    if len(dataframe) > MAX_CHART_POINTS:
        note = f"仅展示前 {MAX_CHART_POINTS} 个数据点"
        detail = f"{spec.title}（{note}）" if spec.title else note
    return (
        spec,
        spec.as_dict(),
        png,
        TraceStep(
            "图表",
            "ok",
            f"已渲染 {spec.type} 图：{spec.x} → {spec.y}",
            detail=detail,
        ),
    )


def _profile_references(
    profile: ProfileReport,
    requests: Sequence[ClarificationRequest],
) -> tuple[ProfileReference, ...]:
    references: list[ProfileReference] = []
    for request in requests:
        matched = [
            anomaly
            for anomaly in profile.anomalies
            if _request_matches_anomaly(request, anomaly)
        ]
        if not matched:
            continue
        references.append(
            ProfileReference(
                code=request.code,
                column=request.column,
                count=sum(anomaly.count for anomaly in matched),
                reason=request.reason,
            )
        )
    return tuple(references)


def _request_matches_anomaly(request: ClarificationRequest, anomaly: Anomaly) -> bool:
    if request.code != anomaly.code:
        return False
    if not request.column or not anomaly.column:
        return True
    columns = {part.strip() for part in request.column.split("/") if part.strip()}
    return anomaly.column in columns


def _merge_references(
    *groups: Sequence[ProfileReference],
) -> tuple[ProfileReference, ...]:
    merged: list[ProfileReference] = []
    seen: set[tuple[str, str | None, str]] = set()
    for group in groups:
        for reference in group:
            key = (reference.code, reference.column, reference.reason)
            if key in seen:
                continue
            seen.add(key)
            merged.append(reference)
    return tuple(merged)


def _emit_step(progress: ProgressSink | None, step: TraceStep) -> None:
    if progress is None:
        return
    progress(
        ProgressEvent(
            kind="step",
            name=step.name,
            status=step.status,
            summary=step.summary,
            detail=step.detail,
        )
    )


def _append_step(
    steps: list[TraceStep],
    step: TraceStep,
    progress: ProgressSink | None,
) -> None:
    steps.append(step)
    _emit_step(progress, step)


def _format_repair_history(history: Sequence[RepairAttempt]) -> str:
    return "\n".join(
        f"- 第 {item.attempt} 次：{item.error}" for item in history
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
