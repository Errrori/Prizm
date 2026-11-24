"""LangGraph 编排：澄清（interrupt）→ 规划 → 执行 → 条件重试 → 图表 → 结论 → 记忆。

图是单一事实源：``Orchestrator`` 只负责数据字典路由与对外 API，分析链路全部在图中。
澄清通过 ``interrupt`` 暂停，由调用方用 ``Command(resume=...)`` 逐项确认后恢复。
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, TypedDict

import duckdb
import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from prizm.agent.clarify import ClarificationOption, ClarificationRequest, ClarifyGate
from prizm.agent.composer import Conclusion, ConclusionComposer, template_conclusion
from prizm.agent.memory import ConversationMemory, Turn
from prizm.agent.planner import Plan, Planner, PlanParseError
from prizm.agent.progress import ProgressEvent, ProgressSink
from prizm.agent.repair import (
    MAX_REPAIR_ATTEMPTS,
    RepairAttempt,
    RepairFailed,
    build_repair_feedback,
    format_repair_error,
)
from prizm.agent.results import Answer, Clarification
from prizm.agent.trace import ProfileReference, Trace, TraceStep, policy_label
from prizm.engine.duckdb_engine import (
    DuckDBEngine,
    QueryTimeoutError,
    SQLValidationError,
)
from prizm.profiling.models import Anomaly, ProfileReport

_RETRYABLE_EXECUTE_ERRORS = (SQLValidationError, QueryTimeoutError, duckdb.Error)
_HISTORY_PREVIEW_ROWS = 3


class AnalysisState(TypedDict, total=False):
    """图状态：只放本轮中间结果，不放全量数据到 prompt。"""

    question: str
    attempt: int
    plan: Plan | None
    sql: str | None
    dataframe: pd.DataFrame | None
    repair_feedback: str | None
    history: list[RepairAttempt]
    error: str | None
    failed: bool
    needs_clarification: bool
    pending_requests: list[ClarificationRequest]
    clarify_index: int
    steps: list[TraceStep]
    references: list[ProfileReference]
    chart: dict[str, Any] | None
    chart_png: bytes | None
    conclusion: str | None
    conclusion_source: str | None
    answer: Answer | None
    started_at: float


@dataclass
class AnalysisDeps:
    """图节点的静态依赖；``progress`` 每次调用前由编排层替换。"""

    planner: Planner
    engine: DuckDBEngine
    frame: pd.DataFrame
    view_name: str
    profile_summary: str | None = None
    composer: ConclusionComposer | None = None
    clarify_gate: ClarifyGate | None = None
    profile: ProfileReport | None = None
    memory: ConversationMemory | None = None
    progress: ProgressSink | None = None
    streaming: bool = False
    max_attempts: int = MAX_REPAIR_ATTEMPTS
    conversation_summary: str | None = None
    confirmed_policies: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or not 1 <= self.max_attempts <= MAX_REPAIR_ATTEMPTS
        ):
            raise ValueError(f"max_attempts 必须在 1 到 {MAX_REPAIR_ATTEMPTS} 之间")


_CHECKPOINT_TYPES = (
    ("prizm.agent.trace", "TraceStep"),
    ("prizm.agent.trace", "ProfileReference"),
    ("prizm.agent.trace", "Trace"),
    ("prizm.agent.planner", "Plan"),
    ("prizm.agent.repair", "RepairAttempt"),
    ("prizm.agent.clarify", "ClarificationRequest"),
    ("prizm.agent.clarify", "ClarificationOption"),
    ("prizm.agent.results", "Answer"),
    ("prizm.agent.results", "Clarification"),
)


# DEBT-029: 图内重试与 Repairer 逻辑重复; checkpointer 为进程内 pickle 兜底。
def default_checkpointer() -> MemorySaver:
    """带 pickle 兜底的进程内 checkpointer：允许暂停时保存 DataFrame 等对象。"""
    return MemorySaver(
        serde=JsonPlusSerializer(
            pickle_fallback=True,
            allowed_msgpack_modules=_CHECKPOINT_TYPES,
        )
    )


def _stream_writer() -> Any | None:
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:
        return None


def _publish(deps: AnalysisDeps, event: ProgressEvent) -> None:
    """有回调时直接回调；处于图级流式模式时写入 LangGraph custom stream。"""
    if deps.progress is not None:
        deps.progress(event)
        return
    if deps.streaming:
        writer = _stream_writer()
        if writer is not None:
            writer(event)


def _emit_start(deps: AnalysisDeps, name: str, summary: str) -> None:
    _publish(deps, ProgressEvent(kind="step_start", name=name, summary=summary))


def _emit_step(
    deps: AnalysisDeps,
    name: str,
    status: str,
    summary: str,
    detail: str | None = None,
) -> None:
    _publish(
        deps,
        ProgressEvent(
            kind="step",
            name=name,
            status=status,
            summary=summary,
            detail=detail,
        ),
    )


def _append_step(state: AnalysisState, step: TraceStep) -> list[TraceStep]:
    return [*state.get("steps", []), step]


def _policies(deps: AnalysisDeps) -> dict[str, str]:
    if deps.clarify_gate is not None:
        return deps.clarify_gate.policy_summary()
    return dict(deps.confirmed_policies or {})


def _conversation_summary(deps: AnalysisDeps) -> str | None:
    if deps.memory is not None:
        return deps.memory.format_for_prompt()
    return deps.conversation_summary


def _elapsed_ms(state: AnalysisState) -> float | None:
    started = state.get("started_at")
    if started is None:
        return None
    return (time.perf_counter() - started) * 1000


def _option_for(request: ClarificationRequest, value: str) -> ClarificationOption:
    text = str(value).strip()
    for option in request.options:
        if text in {option.value, option.label}:
            return option
    raise ValueError(f"无效的澄清选项: {value!r}")


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


def _format_repair_history(history: Sequence[RepairAttempt]) -> str:
    return "\n".join(f"- 第 {item.attempt} 次：{item.error}" for item in history)


def _clarify_check_node(deps: AnalysisDeps):
    def node(state: AnalysisState) -> dict[str, Any]:
        question = state["question"]
        if deps.clarify_gate is None or deps.profile is None:
            step = TraceStep("体检与澄清判定", "ok", "未启用澄清门禁，直接进入规划")
            _emit_step(deps, step.name, step.status, step.summary, step.detail)
            return {
                "needs_clarification": False,
                "pending_requests": [],
                "clarify_index": 0,
                "steps": _append_step(state, step),
            }

        decision = deps.clarify_gate.judge(question, deps.profile)
        requests = list(decision.requests)
        references = list(_profile_references(deps.profile, requests))
        if requests:
            step = TraceStep(
                "体检与澄清判定",
                "clarify",
                f"触发 {len(requests)} 项澄清，暂不执行",
                detail="；".join(request.reason for request in requests),
            )
            _emit_step(deps, step.name, step.status, step.summary, step.detail)
            return {
                "needs_clarification": True,
                "pending_requests": requests,
                "clarify_index": 0,
                "steps": _append_step(state, step),
                "references": [*state.get("references", []), *references],
            }
        step = TraceStep(
            "体检与澄清判定",
            "ok",
            f"体检 {len(deps.profile.anomalies)} 条异常，无新增待澄清项，进入规划",
        )
        _emit_step(deps, step.name, step.status, step.summary, step.detail)
        return {
            "needs_clarification": False,
            "pending_requests": [],
            "clarify_index": 0,
            "steps": _append_step(state, step),
            "references": [*state.get("references", []), *references],
        }

    return node


def _clarify_ask_node(deps: AnalysisDeps):
    def node(state: AnalysisState) -> dict[str, Any]:
        if deps.clarify_gate is None:
            return {}
        requests = state.get("pending_requests", [])
        index = int(state.get("clarify_index", 0))
        if index >= len(requests):
            return {}
        request = requests[index]
        remaining = len(requests) - index - 1
        value = interrupt({"request": request, "remaining": remaining})
        option = _option_for(request, value)
        deps.clarify_gate.confirm(request.code, option.label, question=state["question"])
        step = TraceStep(
            "按确认口径处理",
            "ok",
            f"{policy_label(request.code)}：{option.label}",
            detail=option.description or None,
        )
        _emit_step(deps, step.name, step.status, step.summary, step.detail)
        return {
            "clarify_index": index + 1,
            "steps": _append_step(state, step),
        }

    return node


def _plan_node(deps: AnalysisDeps):
    def node(state: AnalysisState) -> dict[str, Any]:
        attempt = int(state.get("attempt", 0)) + 1
        _emit_start(
            deps,
            "SQL 规划",
            "正在生成 SQL…" if attempt == 1 else f"第 {attempt} 次重新生成 SQL…",
        )
        try:
            plan = deps.planner.plan(
                state["question"],
                deps.view_name,
                deps.frame,
                profile_summary=deps.profile_summary,
                confirmed_policies=_policies(deps) or None,
                conversation_summary=_conversation_summary(deps),
                repair_feedback=state.get("repair_feedback"),
            )
        except PlanParseError as exc:
            error = format_repair_error(exc)
            _emit_step(deps, "SQL 规划", "failed", "SQL 生成失败", error)
            history = [*state.get("history", []), RepairAttempt(attempt, None, error)]
            return {
                "attempt": attempt,
                "plan": None,
                "sql": None,
                "failed": True,
                "error": error,
                "history": history,
                "repair_feedback": build_repair_feedback(
                    attempt=attempt,
                    failed_sql=None,
                    error=error,
                ),
            }

        _emit_step(deps, "SQL 规划", "ok", "已生成只读 SQL", plan.sql)
        return {
            "attempt": attempt,
            "plan": plan,
            "sql": plan.sql,
            "failed": False,
            "error": None,
        }

    return node


def _execute_node(deps: AnalysisDeps):
    def node(state: AnalysisState) -> dict[str, Any]:
        plan = state.get("plan")
        if plan is None:
            return {"failed": True, "error": "缺少可执行计划"}
        attempt = int(state.get("attempt", 1))
        _emit_start(deps, "执行", "正在执行 SQL…")
        try:
            dataframe = deps.engine.execute(plan.sql)
        except _RETRYABLE_EXECUTE_ERRORS as exc:
            error = format_repair_error(exc)
            _emit_step(deps, "执行", "failed", "SQL 执行失败", error)
            history = [*state.get("history", []), RepairAttempt(attempt, plan.sql, error)]
            return {
                "dataframe": None,
                "failed": True,
                "error": error,
                "history": history,
                "repair_feedback": build_repair_feedback(
                    attempt=attempt,
                    failed_sql=plan.sql,
                    error=error,
                ),
            }

        steps = _append_step(state, TraceStep("SQL 规划", "ok", "已生成只读 SQL", plan.sql))
        if attempt > 1:
            repaired = TraceStep(
                "自修复",
                "repaired",
                f"第 {attempt} 次尝试成功",
                detail=_format_repair_history(state.get("history", [])) or None,
            )
            steps.append(repaired)
            _emit_step(
                deps,
                repaired.name,
                repaired.status,
                repaired.summary,
                repaired.detail,
            )
        executed = TraceStep(
            "执行",
            "ok",
            f"返回 {len(dataframe)} 行 × {len(dataframe.columns)} 列",
        )
        steps.append(executed)
        _emit_step(deps, executed.name, executed.status, executed.summary)
        return {
            "dataframe": dataframe,
            "failed": False,
            "error": None,
            "steps": steps,
        }

    return node


def _chart_node(deps: AnalysisDeps):
    def node(state: AnalysisState) -> dict[str, Any]:
        plan = state.get("plan")
        dataframe = state.get("dataframe")
        if plan is None or plan.chart is None or dataframe is None:
            return {}

        from prizm.viz.charts import (
            MAX_CHART_POINTS,
            ChartError,
            parse_chart_spec,
            render_chart,
        )

        try:
            spec = parse_chart_spec(plan.chart, dataframe.columns)
            if spec is None:
                return {}
            png = render_chart(spec, dataframe)
        except ChartError as exc:
            step = TraceStep(
                "图表",
                "skipped",
                "图表参数不可用，已改用表格展示",
                detail=str(exc),
            )
            _emit_step(deps, step.name, step.status, step.summary, step.detail)
            return {"steps": _append_step(state, step)}

        detail = spec.title
        if len(dataframe) > MAX_CHART_POINTS:
            note = f"仅展示前 {MAX_CHART_POINTS} 个数据点"
            detail = f"{spec.title}（{note}）" if spec.title else note
        step = TraceStep(
            "图表",
            "ok",
            f"已渲染 {spec.type} 图：{spec.x} → {spec.y}",
            detail=detail,
        )
        _emit_step(deps, step.name, step.status, step.summary, step.detail)
        return {
            "chart": spec.as_dict(),
            "chart_png": png,
            "steps": _append_step(state, step),
        }

    return node


def _conclude_node(deps: AnalysisDeps):
    def node(state: AnalysisState) -> dict[str, Any]:
        dataframe = state.get("dataframe")
        if dataframe is None:
            raise RuntimeError("结论生成缺少结果数据")

        if deps.composer is None:
            conclusion = Conclusion(template_conclusion(dataframe))
            step = TraceStep("结论生成", "ok", "使用模板结论（未启用模型归纳）")
            _emit_step(deps, step.name, step.status, step.summary, step.detail)
            return {
                "conclusion": conclusion.text,
                "conclusion_source": conclusion.source,
                "steps": _append_step(state, step),
            }

        _emit_start(deps, "结论生成", "正在归纳结论…")
        token_callback = None
        if deps.progress is not None or deps.streaming:

            def _emit_token(chunk: str) -> None:
                _publish(deps, ProgressEvent(kind="conclusion_token", text=chunk))

            token_callback = _emit_token

        conclusion = deps.composer.compose(
            state["question"],
            str(state.get("sql") or ""),
            dataframe,
            on_token=token_callback,
        )
        if conclusion.source == "llm":
            step = TraceStep("结论生成", "ok", "已用模型归纳结果", detail=conclusion.text)
        else:
            step = TraceStep(
                "结论生成",
                "fallback",
                "模型结论不可用，已回退模板",
                detail=conclusion.fallback_reason,
            )
        _emit_step(deps, step.name, step.status, step.summary, step.detail)
        return {
            "conclusion": conclusion.text,
            "conclusion_source": conclusion.source,
            "steps": _append_step(state, step),
        }

    return node


def _finalize_node(deps: AnalysisDeps):
    def node(state: AnalysisState) -> dict[str, Any]:
        dataframe = state.get("dataframe")
        if dataframe is None:
            raise RuntimeError("最终结果缺少数据")

        attempt = int(state.get("attempt", 1))
        trace = Trace(
            question=state["question"],
            steps=tuple(state.get("steps", [])),
            generated_sql=state.get("sql"),
            execution_status="repaired" if attempt > 1 else "succeeded",
            profile_references=tuple(state.get("references", [])),
            confirmed_policies=tuple(_policies(deps).items()),
            attempts=attempt,
            elapsed_ms=_elapsed_ms(state),
        )
        answer = Answer(
            question=state["question"],
            sql=str(state.get("sql") or ""),
            dataframe=dataframe,
            chart=state.get("chart"),
            chart_png=state.get("chart_png"),
            conclusion=str(state.get("conclusion") or ""),
            conclusion_source=str(state.get("conclusion_source") or "template"),
            attempts=attempt,
            repair_history=tuple(state.get("history", [])),
            trace=trace,
        )
        if deps.memory is not None:
            deps.memory.add(
                Turn(
                    question=answer.question,
                    sql=answer.sql,
                    columns=tuple(str(column) for column in dataframe.columns),
                    row_count=len(dataframe),
                    preview=tuple(
                        {str(key): value for key, value in row.items()}
                        for row in dataframe.head(_HISTORY_PREVIEW_ROWS).to_dict(
                            orient="records"
                        )
                    ),
                    chart=answer.chart,
                )
            )
        return {"answer": answer}

    return node


def _failed_node(deps: AnalysisDeps):
    def node(state: AnalysisState) -> dict[str, Any]:
        return {**state, "failed": True}

    return node


def _route_after_check(state: AnalysisState) -> str:
    return "clarify_ask" if state.get("needs_clarification") else "plan"


def _route_after_ask(state: AnalysisState) -> str:
    pending = state.get("pending_requests", [])
    index = int(state.get("clarify_index", 0))
    return "clarify_ask" if index < len(pending) else "plan"


def _route_after_plan(deps: AnalysisDeps):
    def route(state: AnalysisState) -> str:
        if not state.get("failed") and state.get("plan") is not None:
            return "execute"
        if int(state.get("attempt", 0)) < deps.max_attempts:
            return "plan"
        return "failed"

    return route


def _route_after_execute(deps: AnalysisDeps):
    def route(state: AnalysisState) -> str:
        if not state.get("failed"):
            return "chart"
        if int(state.get("attempt", 0)) < deps.max_attempts:
            return "plan"
        return "failed"

    return route


def build_analysis_graph(
    deps: AnalysisDeps,
    *,
    memory: ConversationMemory | None = None,
    checkpointer: Any | None = None,
):
    """构建并编译分析状态图；澄清需要 ``checkpointer`` 才能中断/恢复。"""
    if memory is not None:
        deps = replace(deps, memory=memory)
    builder = StateGraph(AnalysisState)
    builder.add_node("clarify_check", _clarify_check_node(deps))
    builder.add_node("clarify_ask", _clarify_ask_node(deps))
    builder.add_node("plan", _plan_node(deps))
    builder.add_node("execute", _execute_node(deps))
    builder.add_node("chart", _chart_node(deps))
    builder.add_node("conclude", _conclude_node(deps))
    builder.add_node("finalize", _finalize_node(deps))
    builder.add_node("failed", _failed_node(deps))

    builder.add_edge(START, "clarify_check")
    builder.add_conditional_edges(
        "clarify_check",
        _route_after_check,
        {"clarify_ask": "clarify_ask", "plan": "plan"},
    )
    builder.add_conditional_edges(
        "clarify_ask",
        _route_after_ask,
        {"clarify_ask": "clarify_ask", "plan": "plan"},
    )
    builder.add_conditional_edges(
        "plan",
        _route_after_plan(deps),
        {"execute": "execute", "plan": "plan", "failed": "failed"},
    )
    builder.add_conditional_edges(
        "execute",
        _route_after_execute(deps),
        {"chart": "chart", "plan": "plan", "failed": "failed"},
    )
    builder.add_edge("chart", "conclude")
    builder.add_edge("conclude", "finalize")
    builder.add_edge("finalize", END)
    builder.add_edge("failed", END)
    return builder.compile(checkpointer=checkpointer)


def build_trace_from_state(
    deps: AnalysisDeps,
    state: Mapping[str, Any],
    *,
    status: str,
    question: str,
) -> Trace:
    return Trace(
        question=question,
        steps=tuple(state.get("steps", [])),
        generated_sql=state.get("sql"),
        execution_status=status,
        profile_references=tuple(state.get("references", [])),
        confirmed_policies=tuple(_policies(deps).items()),
        attempts=int(state.get("attempt", 0) or 0),
        elapsed_ms=_elapsed_ms(state),
    )


def build_failed_trace(
    deps: AnalysisDeps,
    state: Mapping[str, Any],
    *,
    question: str,
) -> Trace:
    history = tuple(state.get("history", ()))
    attempts = int(state.get("attempt", len(history) or 1))
    steps = list(state.get("steps", []))
    steps.append(
        TraceStep(
            "SQL 规划与执行",
            "failed",
            f"已尝试 {attempts} 次仍未成功",
            detail=_format_repair_history(history) or str(state.get("error") or ""),
        )
    )
    return Trace(
        question=question,
        steps=tuple(steps),
        generated_sql=state.get("sql"),
        execution_status="failed",
        profile_references=tuple(state.get("references", [])),
        confirmed_policies=tuple(_policies(deps).items()),
        attempts=attempts,
        elapsed_ms=_elapsed_ms(state),
    )


def repair_failed_from_state(
    question: str,
    state: Mapping[str, Any],
) -> RepairFailed:
    history = tuple(state.get("history", ()))
    return RepairFailed(
        question=question,
        attempts=int(state.get("attempt", len(history) or 1)),
        last_error=str(state.get("error") or "未知错误"),
        history=history,
    )


def initial_state(question: str) -> AnalysisState:
    return {
        "question": question,
        "attempt": 0,
        "history": [],
        "steps": [],
        "references": [],
        "failed": False,
        "needs_clarification": False,
        "pending_requests": [],
        "clarify_index": 0,
        "started_at": time.perf_counter(),
    }


def run_analysis_graph(
    question: str,
    *,
    deps: AnalysisDeps,
    memory: ConversationMemory | None = None,
) -> Answer | Clarification:
    """单轮执行图（不跨轮恢复）；命中澄清时返回 Clarification。"""
    if not question.strip():
        raise ValueError("问题不能为空")
    checkpointer = default_checkpointer()
    graph = build_analysis_graph(deps, memory=memory, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "run"}}
    result = graph.invoke(initial_state(question.strip()), config)
    state = graph.get_state(config).values
    interrupts = result.get("__interrupt__") or ()
    if interrupts:
        payload = interrupts[0].value
        return Clarification(
            question=question.strip(),
            request=payload["request"],
            remaining=int(payload.get("remaining", 0)),
            trace=build_trace_from_state(
                deps, state, status="not_run", question=question.strip()
            ),
        )
    if state.get("failed"):
        raise repair_failed_from_state(question.strip(), state)
    answer = state.get("answer")
    if answer is None:
        raise RuntimeError("分析图未产出结果")
    return answer


__all__ = [
    "AnalysisDeps",
    "AnalysisState",
    "build_analysis_graph",
    "build_failed_trace",
    "build_trace_from_state",
    "default_checkpointer",
    "initial_state",
    "repair_failed_from_state",
    "run_analysis_graph",
]
