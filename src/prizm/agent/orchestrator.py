"""对话式编排：数据字典路由 + LangGraph 分析链路（澄清走 interrupt/resume）。

对外 API（``ask`` / ``resolve`` / ``last_trace`` / ``pending_clarification``）保持不变，
内部把规划、执行、条件重试、图表、结论、记忆全部交给 ``agent.graph``。
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from langgraph.types import Command

from prizm.agent.clarify import ClarificationOption, ClarificationRequest, ClarifyGate
from prizm.agent.composer import ConclusionComposer
from prizm.agent.graph import (
    AnalysisDeps,
    build_analysis_graph,
    build_failed_trace,
    build_trace_from_state,
    default_checkpointer,
    initial_state,
    repair_failed_from_state,
)
from prizm.agent.memory import ConversationMemory, Turn
from prizm.agent.planner import Planner
from prizm.agent.progress import ProgressEvent, ProgressSink
from prizm.agent.repair import MAX_REPAIR_ATTEMPTS, Repairer
from prizm.agent.results import Answer, Clarification, format_result_table
from prizm.agent.trace import Trace, TraceStep
from prizm.datasource.base import BaseConnector
from prizm.engine.duckdb_engine import DuckDBEngine
from prizm.profiling.models import ProfileReport
from prizm.profiling.profiler import ProfileOptions, profile_frame
from prizm.rag.retriever import DictionaryAnswer, DictionaryQA, is_dictionary_question


@dataclass(frozen=True, slots=True)
class _Pending:
    question: str
    request: ClarificationRequest
    remaining: int


class Orchestrator:
    """单数据集、多轮问答；上传后立即体检，分析链路由图编排。"""

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
        self._pending: _Pending | None = None
        self._last_trace: Trace | None = None
        self._thread_id: str | None = None
        self._turn_seq = 0
        self._instance_id = uuid.uuid4().hex

        max_attempts = repairer.max_attempts if repairer is not None else MAX_REPAIR_ATTEMPTS
        self._deps = AnalysisDeps(
            planner=planner,
            engine=engine,
            frame=self._frame,
            view_name=self.view_name,
            profile_summary=self._profile.to_prompt(
                max_chars=self._profile_options.max_prompt_chars
            ),
            composer=conclusion_composer,
            clarify_gate=self._clarify_gate,
            profile=self._profile,
            memory=self._memory,
            max_attempts=max_attempts,
        )
        self._checkpointer = default_checkpointer()
        self._graph = build_analysis_graph(self._deps, checkpointer=self._checkpointer)

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
        return Clarification(
            question=self._pending.question,
            request=self._pending.request,
            remaining=self._pending.remaining,
            trace=self._last_trace,
        )

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
            if progress is not None:
                for step in steps:
                    progress(
                        ProgressEvent(
                            kind="step",
                            name=step.name,
                            status=step.status,
                            summary=step.summary,
                            detail=step.detail,
                        )
                    )
            return answer
        return self._run(text, progress)

    def resolve(
        self,
        value: str,
        *,
        progress: ProgressSink | None = None,
    ) -> Answer | Clarification:
        """记录用户对当前澄清项的回复；全部澄清完成后恢复图并执行。"""
        pending = self._pending
        if pending is None:
            raise RuntimeError("当前没有待澄清的问题")
        if self._thread_id is None:
            raise RuntimeError("当前没有待澄清的问题")
        option = _match_option(pending.request, value)
        config = {"configurable": {"thread_id": self._thread_id}}
        snapshot = self._invoke(Command(resume=option.value), config, progress)
        return self._handle(snapshot, pending.question)

    def _run(
        self,
        question: str,
        progress: ProgressSink | None,
    ) -> Answer | Clarification:
        self._turn_seq += 1
        self._thread_id = f"{self._instance_id}:{self._turn_seq}"
        config = {"configurable": {"thread_id": self._thread_id}}
        snapshot = self._invoke(initial_state(question), config, progress)
        return self._handle(snapshot, question)

    def _invoke(
        self,
        payload: object,
        config: dict,
        progress: ProgressSink | None,
    ):
        """有进度回调时走 LangGraph custom stream，否则直接 invoke。"""
        self._deps.progress = None
        self._deps.streaming = progress is not None
        try:
            if progress is None:
                self._graph.invoke(payload, config)
            else:
                for chunk in self._graph.stream(payload, config, stream_mode="custom"):
                    if isinstance(chunk, ProgressEvent):
                        progress(chunk)
        finally:
            self._deps.streaming = False
        return self._graph.get_state(config)

    def _handle(
        self,
        snapshot: object,
        question: str,
    ) -> Answer | Clarification:
        state = snapshot.values
        interrupts = getattr(snapshot, "interrupts", ()) or ()
        if interrupts:
            payload = interrupts[0].value
            request = payload["request"]
            remaining = int(payload.get("remaining", 0))
            self._pending = _Pending(question, request, remaining)
            trace = build_trace_from_state(
                self._deps, state, status="not_run", question=question
            )
            self._last_trace = trace
            return Clarification(
                question=question,
                request=request,
                remaining=remaining,
                trace=trace,
            )

        self._pending = None
        if state.get("failed"):
            self._last_trace = build_failed_trace(self._deps, state, question=question)
            raise repair_failed_from_state(question, state)

        answer = state.get("answer")
        if answer is None:
            raise RuntimeError("编排未产出结果")
        self._last_trace = answer.trace
        return answer


def _match_option(request: ClarificationRequest, value: str) -> ClarificationOption:
    text = value.strip()
    for option in request.options:
        if text in {option.value, option.label}:
            return option
    raise ValueError(f"无效的澄清选项: {value!r}")


__all__ = ["Answer", "Clarification", "Orchestrator", "format_result_table"]
