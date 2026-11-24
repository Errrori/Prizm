"""LangGraph 编排骨架：把“规划 → 执行 → 条件重试 → 图表 → 结论”建成显式状态图。

本模块暂不替换 ``Orchestrator``，只提供可独立运行、可单测的图结构，
对外仍产出与旧编排一致的 ``Answer`` / ``Clarification``。
接入 Orchestrator、HITL ``interrupt`` 与 checkpointer 留待下一任务（见 DEBT-028）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

import duckdb
import pandas as pd
from langgraph.graph import END, START, StateGraph

from prizm.agent.clarify import ClarifyGate
from prizm.agent.composer import Conclusion, ConclusionComposer, template_conclusion
from prizm.agent.memory import ConversationMemory, Turn
from prizm.agent.orchestrator import Answer, Clarification
from prizm.agent.planner import Plan, Planner, PlanParseError
from prizm.agent.progress import ProgressEvent, ProgressSink
from prizm.agent.repair import (
    MAX_REPAIR_ATTEMPTS,
    RepairAttempt,
    RepairFailed,
    build_repair_feedback,
    format_repair_error,
)
from prizm.agent.trace import Trace, TraceStep
from prizm.engine.duckdb_engine import (
    DuckDBEngine,
    QueryTimeoutError,
    SQLValidationError,
)
from prizm.profiling.models import ProfileReport

# DEBT-028: 骨架阶段重试边与 Repairer 逻辑重复, 且未接入 interrupt/checkpointer/流式。
_RETRYABLE_EXECUTE_ERRORS = (SQLValidationError, QueryTimeoutError, duckdb.Error)
_HISTORY_PREVIEW_ROWS = 3


class AnalysisState(TypedDict, total=False):
    """图的状态：只放本轮分析的中间结果，不放全量历史数据。"""

    question: str
    attempt: int
    plan: Plan | None
    sql: str | None
    dataframe: pd.DataFrame | None
    repair_feedback: str | None
    history: list[RepairAttempt]
    error: str | None
    failed: bool
    chart: dict[str, Any] | None
    chart_png: bytes | None
    conclusion: str | None
    conclusion_source: str | None
    confirmed_policies: dict[str, str] | None
    clarification: Clarification | None
    answer: Answer | None
    steps: list[TraceStep]


@dataclass
class AnalysisDeps:
    """图节点依赖的静态上下文；动态结果走 ``AnalysisState``。"""

    planner: Planner
    engine: DuckDBEngine
    frame: pd.DataFrame
    view_name: str
    profile_summary: str | None = None
    confirmed_policies: Mapping[str, str] | None = None
    conversation_summary: str | None = None
    composer: ConclusionComposer | None = None
    clarify_gate: ClarifyGate | None = None
    profile: ProfileReport | None = None
    progress: ProgressSink | None = None
    max_attempts: int = MAX_REPAIR_ATTEMPTS

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or not 1 <= self.max_attempts <= MAX_REPAIR_ATTEMPTS
        ):
            raise ValueError(f"max_attempts 必须在 1 到 {MAX_REPAIR_ATTEMPTS} 之间")


def _emit_start(progress: ProgressSink | None, name: str, summary: str) -> None:
    if progress is not None:
        progress(ProgressEvent(kind="step_start", name=name, summary=summary))


def _emit_step(
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


def _append_step(state: AnalysisState, step: TraceStep) -> list[TraceStep]:
    return [*state.get("steps", []), step]


def _clarify_node(deps: AnalysisDeps):
    def node(state: AnalysisState) -> dict[str, Any]:
        question = state["question"]
        if deps.clarify_gate is None or deps.profile is None:
            step = TraceStep("体检与澄清判定", "ok", "未启用澄清门禁，直接进入规划")
            return {
                "steps": _append_step(state, step),
                "confirmed_policies": dict(deps.confirmed_policies or {}) or None,
            }
        decision = deps.clarify_gate.judge(question, deps.profile)
        if decision.needs_clarification:
            step = TraceStep(
                "体检与澄清判定",
                "clarify",
                f"触发 {len(decision.requests)} 项澄清，暂不执行",
                detail="；".join(request.reason for request in decision.requests),
            )
            _emit_step(deps.progress, step.name, step.status, step.summary, step.detail)
            clarification = Clarification(
                question=question,
                request=decision.requests[0],
                remaining=len(decision.requests) - 1,
            )
            return {
                "steps": _append_step(state, step),
                "clarification": clarification,
            }
        step = TraceStep(
            "体检与澄清判定",
            "ok",
            f"体检 {len(deps.profile.anomalies)} 条异常，无新增待澄清项，进入规划",
        )
        _emit_step(deps.progress, step.name, step.status, step.summary, step.detail)
        policies = deps.clarify_gate.policy_summary()
        return {
            "steps": _append_step(state, step),
            "confirmed_policies": policies or None,
        }

    return node


def _plan_node(deps: AnalysisDeps):
    def node(state: AnalysisState) -> dict[str, Any]:
        attempt = int(state.get("attempt", 0)) + 1
        _emit_start(
            deps.progress,
            "SQL 规划",
            "正在生成 SQL…" if attempt == 1 else f"第 {attempt} 次重新生成 SQL…",
        )
        policies = state.get("confirmed_policies") or deps.confirmed_policies
        try:
            plan = deps.planner.plan(
                state["question"],
                deps.view_name,
                deps.frame,
                profile_summary=deps.profile_summary,
                confirmed_policies=policies,
                conversation_summary=deps.conversation_summary,
                repair_feedback=state.get("repair_feedback"),
            )
        except PlanParseError as exc:
            error = format_repair_error(exc)
            step = TraceStep("SQL 规划", "failed", "SQL 生成失败", detail=error)
            _emit_step(deps.progress, step.name, step.status, step.summary, step.detail)
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
                "steps": _append_step(state, step),
            }

        step = TraceStep("SQL 规划", "ok", "已生成只读 SQL", detail=plan.sql)
        _emit_step(deps.progress, step.name, step.status, step.summary, step.detail)
        return {
            "attempt": attempt,
            "plan": plan,
            "sql": plan.sql,
            "failed": False,
            "error": None,
            "steps": _append_step(state, step),
        }

    return node


def _execute_node(deps: AnalysisDeps):
    def node(state: AnalysisState) -> dict[str, Any]:
        plan = state.get("plan")
        if plan is None:
            step = TraceStep("执行", "failed", "缺少可执行计划")
            return {"failed": True, "error": "缺少可执行计划", "steps": _append_step(state, step)}
        _emit_start(deps.progress, "执行", "正在执行 SQL…")
        try:
            dataframe = deps.engine.execute(plan.sql)
        except _RETRYABLE_EXECUTE_ERRORS as exc:
            error = format_repair_error(exc)
            attempt = int(state.get("attempt", 1))
            step = TraceStep("执行", "failed", "SQL 执行失败", detail=error)
            _emit_step(deps.progress, step.name, step.status, step.summary, step.detail)
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
                "steps": _append_step(state, step),
            }

        step = TraceStep(
            "执行",
            "ok",
            f"返回 {len(dataframe)} 行 × {len(dataframe.columns)} 列",
        )
        _emit_step(deps.progress, step.name, step.status, step.summary, step.detail)
        return {
            "dataframe": dataframe,
            "failed": False,
            "error": None,
            "steps": _append_step(state, step),
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
            _emit_step(deps.progress, step.name, step.status, step.summary, step.detail)
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
        _emit_step(deps.progress, step.name, step.status, step.summary, step.detail)
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
            _emit_step(deps.progress, step.name, step.status, step.summary, step.detail)
            return {
                "conclusion": conclusion.text,
                "conclusion_source": conclusion.source,
                "steps": _append_step(state, step),
            }

        _emit_start(deps.progress, "结论生成", "正在归纳结论…")
        conclusion = deps.composer.compose(
            state["question"],
            str(state.get("sql") or ""),
            dataframe,
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
        _emit_step(deps.progress, step.name, step.status, step.summary, step.detail)
        return {
            "conclusion": conclusion.text,
            "conclusion_source": conclusion.source,
            "steps": _append_step(state, step),
        }

    return node


def _finalize_node(deps: AnalysisDeps, memory: ConversationMemory | None):
    def node(state: AnalysisState) -> dict[str, Any]:
        dataframe = state.get("dataframe")
        if dataframe is None:
            raise RuntimeError("最终结果缺少数据")

        attempt = int(state.get("attempt", 1))
        history = tuple(state.get("history", []))
        steps = list(state.get("steps", []))
        trace = Trace(
            question=state["question"],
            steps=tuple(steps),
            generated_sql=state.get("sql"),
            execution_status="repaired" if attempt > 1 else "succeeded",
            attempts=attempt,
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
            repair_history=history,
            trace=trace,
        )
        if memory is not None:
            memory.add(
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
        history = tuple(state.get("history", []))
        attempts = int(state.get("attempt", len(history) or 1))
        raise RepairFailed(
            question=state.get("question", ""),
            attempts=attempts,
            last_error=str(state.get("error") or "未知错误"),
            history=history,
        )

    return node


def _route_after_clarify(state: AnalysisState) -> str:
    return "end" if state.get("clarification") is not None else "plan"


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
):
    """构建并编译分析状态图（无 checkpointer，单轮 invoke）。"""
    builder = StateGraph(AnalysisState)
    builder.add_node("clarify", _clarify_node(deps))
    builder.add_node("plan", _plan_node(deps))
    builder.add_node("execute", _execute_node(deps))
    builder.add_node("chart", _chart_node(deps))
    builder.add_node("conclude", _conclude_node(deps))
    builder.add_node("finalize", _finalize_node(deps, memory))
    builder.add_node("failed", _failed_node(deps))

    builder.add_edge(START, "clarify")
    builder.add_conditional_edges("clarify", _route_after_clarify, {"plan": "plan", "end": END})
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
    return builder.compile()


def run_analysis_graph(
    question: str,
    *,
    deps: AnalysisDeps,
    memory: ConversationMemory | None = None,
) -> Answer | Clarification:
    """单轮执行分析图；命中澄清时返回 Clarification，不进入规划。"""
    if not question.strip():
        raise ValueError("问题不能为空")
    graph = build_analysis_graph(deps, memory=memory)
    state = graph.invoke(
        {
            "question": question.strip(),
            "attempt": 0,
            "history": [],
            "steps": [],
        }
    )
    clarification = state.get("clarification")
    if clarification is not None:
        return clarification
    answer = state.get("answer")
    if answer is None:
        raise RuntimeError("分析图未产出结果")
    return answer


__all__ = [
    "AnalysisDeps",
    "AnalysisState",
    "build_analysis_graph",
    "run_analysis_graph",
]
