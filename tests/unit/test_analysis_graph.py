"""LangGraph 编排骨架单测：用假 Planner/Engine，不触网、不真执行 SQL。"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from prizm.agent.clarify import ClarificationOption, ClarificationRequest, ClarifyDecision
from prizm.agent.graph import AnalysisDeps, build_analysis_graph, run_analysis_graph
from prizm.agent.memory import ConversationMemory
from prizm.agent.orchestrator import Answer, Clarification
from prizm.agent.planner import Plan, PlanParseError
from prizm.agent.repair import RepairFailed


class _Planner:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []

    def plan(self, question, view_name, frame, **kwargs):
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _Engine:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls: list[str] = []

    def execute(self, sql):
        self.calls.append(sql)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _StubGate:
    def judge(self, question, profile):
        return ClarifyDecision(
            (
                ClarificationRequest(
                    code="exact_duplicate_rows",
                    reason="存在完全重复行",
                    question="如何处理重复行？",
                    options=(ClarificationOption("keep_all", "保留", ""),),
                ),
            )
        )

    def policy_summary(self):
        return {}


class _StubProfile:
    anomalies: tuple = ()


def _deps(planner, engine, *, frame=None, **kwargs) -> AnalysisDeps:
    frame = frame if frame is not None else pd.DataFrame({"month": ["2024-01"], "sales": [10]})
    return AnalysisDeps(planner=planner, engine=engine, frame=frame, view_name="v", **kwargs)


def test_graph_success_path() -> None:
    frame = pd.DataFrame({"month": ["2024-01"], "sales": [10]})
    planner = _Planner([Plan("SELECT 1")])
    engine = _Engine([frame])

    answer = run_analysis_graph("总销售额是多少", deps=_deps(planner, engine))

    assert isinstance(answer, Answer)
    assert answer.attempts == 1
    assert answer.repaired is False
    assert answer.sql == "SELECT 1"
    assert answer.dataframe.equals(frame)
    assert answer.conclusion
    assert answer.trace is not None
    assert answer.trace.execution_status == "succeeded"


def test_graph_retries_execute_error_through_conditional_edge() -> None:
    frame = pd.DataFrame({"sales": [1]})
    planner = _Planner([Plan("bad sql"), Plan("good sql")])
    engine = _Engine([duckdb.Error("boom"), frame])

    answer = run_analysis_graph("q", deps=_deps(planner, engine, max_attempts=3))

    assert answer.attempts == 2
    assert answer.repaired is True
    assert engine.calls == ["bad sql", "good sql"]
    assert planner.calls[0]["repair_feedback"] is None
    assert "第 1 次尝试失败" in planner.calls[1]["repair_feedback"]


def test_graph_retries_plan_parse_error() -> None:
    frame = pd.DataFrame({"sales": [1]})
    planner = _Planner([PlanParseError("bad json"), Plan("SELECT 1")])
    engine = _Engine([frame])

    answer = run_analysis_graph("q", deps=_deps(planner, engine))

    assert answer.attempts == 2
    assert planner.calls[1]["repair_feedback"]
    assert len(answer.repair_history) == 1


def test_graph_raises_repair_failed_after_max_attempts() -> None:
    planner = _Planner([Plan("a"), Plan("b")])
    engine = _Engine([duckdb.Error("e1"), duckdb.Error("e2")])

    with pytest.raises(RepairFailed) as excinfo:
        run_analysis_graph("q", deps=_deps(planner, engine, max_attempts=2))

    assert excinfo.value.attempts == 2
    assert len(excinfo.value.history) == 2


def test_graph_clarify_short_circuits_before_planning() -> None:
    planner = _Planner([])
    engine = _Engine([])
    deps = _deps(planner, engine, clarify_gate=_StubGate(), profile=_StubProfile())

    result = run_analysis_graph("销售额是多少", deps=deps)

    assert isinstance(result, Clarification)
    assert result.request.code == "exact_duplicate_rows"
    assert result.remaining == 0
    assert planner.calls == []
    assert engine.calls == []


def test_graph_chart_error_is_non_fatal() -> None:
    frame = pd.DataFrame({"sales": [1, 2]})
    planner = _Planner([Plan("SELECT 1", chart={"type": "bar", "x": "missing", "y": "sales"})])
    engine = _Engine([frame])

    answer = run_analysis_graph("q", deps=_deps(planner, engine))

    assert isinstance(answer, Answer)
    assert answer.chart is None
    assert answer.chart_png is None
    assert any(step.name == "图表" and step.status == "skipped" for step in answer.trace.steps)


def test_graph_records_memory_turn() -> None:
    frame = pd.DataFrame({"sales": [1]})
    memory = ConversationMemory()

    run_analysis_graph(
        "q",
        deps=_deps(_Planner([Plan("SELECT 1")]), _Engine([frame])),
        memory=memory,
    )

    assert len(memory.turns) == 1
    assert memory.turns[0].sql == "SELECT 1"


def test_graph_builds_compiled_graph() -> None:
    graph = build_analysis_graph(_deps(_Planner([]), _Engine([])))
    assert hasattr(graph, "invoke")


def test_deps_rejects_invalid_max_attempts() -> None:
    with pytest.raises(ValueError):
        _deps(_Planner([]), _Engine([]), max_attempts=4)
