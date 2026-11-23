import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest

from prizm.agent.composer import ConclusionComposer
from prizm.agent.orchestrator import (
    Answer,
    Clarification,
    Orchestrator,
    format_result_table,
)
from prizm.agent.planner import Planner
from prizm.agent.repair import RepairFailed
from prizm.agent.trace import Trace
from prizm.datasource.csv_connector import CsvConnector
from prizm.engine.duckdb_engine import DuckDBEngine
from prizm.llm.adapter import Completion, LLMError, Message, MockLLM
from prizm.rag.ingest import DictionaryEntry
from prizm.rag.retriever import DataDictionaryRetriever, DictionaryAnswer, DictionaryQA


class CountingConnector(CsvConnector):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.read_count = 0

    def _read(self) -> pd.DataFrame:
        self.read_count += 1
        return super()._read()


def write_orders(path: Path) -> None:
    pd.DataFrame(
        {
            "channel": ["线上", "线上", "线下"],
            "status": ["已完成", "已取消", "已完成"],
            "sales": [100, 999, 200],
        }
    ).to_csv(path, index=False, encoding="utf-8")


def write_dirty_orders(path: Path) -> None:
    pd.DataFrame(
        {
            "order_id": [1, 1, 2, 2, 3],
            "channel": ["线上", "线上", "线下", "线下", "线上"],
            "quantity": [1, None, 2, 2, 4],
            "sales": ["1,200", "800", "800", "800", "300"],
        }
    ).to_csv(path, index=False, encoding="utf-8")


@dataclass
class SequenceLLM:
    responses: list[str]
    calls: list[list[Message]] = field(default_factory=list)

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> Completion:
        self.calls.append([dict(message) for message in messages])
        if not self.responses:
            raise AssertionError("没有更多脚本化响应")
        return Completion(self.responses.pop(0))


@dataclass
class FailingLLM:
    error: Exception

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> Completion:
        raise self.error



def test_orchestrator_runs_plan_end_to_end(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    write_orders(path)
    connector = CountingConnector(path)
    llm = MockLLM(
        json.dumps(
            {
                "sql": ("SELECT SUM(sales) AS total FROM orders WHERE status = '已完成'"),
                "chart": None,
            }
        )
    )

    with DuckDBEngine() as engine:
        orchestrator = Orchestrator(connector, engine, Planner(llm))
        answer = orchestrator.ask("已完成订单的销售额是多少？")

    assert connector.read_count == 1
    assert int(answer.dataframe.loc[0, "total"]) == 300
    assert answer.sql.startswith("SELECT")
    assert answer.question == "已完成订单的销售额是多少？"
    assert answer.empty is False
    assert llm.calls

    assert isinstance(answer.trace, Trace)
    assert answer.trace.execution_status == "succeeded"
    assert answer.trace.generated_sql == answer.sql
    assert [step.name for step in answer.trace.steps] == [
        "体检与澄清判定",
        "SQL 规划",
        "执行",
        "结论生成",
    ]
    assert answer.conclusion
    assert answer.conclusion_source == "template"
    assert orchestrator.last_trace is answer.trace


def test_orchestrator_repairs_after_sql_error(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    write_orders(path)
    bad_sql = "SELECT missing_column FROM orders"
    good_sql = "SELECT SUM(sales) AS total FROM orders WHERE status = '已完成'"
    llm = SequenceLLM(
        [
            json.dumps({"sql": bad_sql, "chart": None}),
            json.dumps({"sql": good_sql, "chart": None}),
        ]
    )

    with DuckDBEngine() as engine:
        orchestrator = Orchestrator(CsvConnector(path), engine, Planner(llm))
        answer = orchestrator.ask("已完成订单的销售额是多少？")

    assert isinstance(answer, Answer)
    assert answer.attempts == 2
    assert answer.repaired is True
    assert len(answer.repair_history) == 1
    assert answer.repair_history[0].sql == bad_sql
    assert "missing_column" in answer.repair_history[0].error
    assert int(answer.dataframe.loc[0, "total"]) == 300
    assert "上一次尝试失败" in llm.calls[1][1]["content"]

    assert isinstance(answer.trace, Trace)
    assert answer.trace.execution_status == "repaired"
    assert answer.trace.attempts == 2
    assert [step.name for step in answer.trace.steps][-3:] == [
        "自修复",
        "执行",
        "结论生成",
    ]


def test_orchestrator_reuses_dataset_and_injects_history(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    write_orders(path)
    sql = "SELECT SUM(sales) AS total FROM orders WHERE status = '已完成'"
    llm = SequenceLLM(
        [
            json.dumps({"sql": sql, "chart": None}),
            json.dumps({"sql": sql, "chart": None}),
        ]
    )
    connector = CountingConnector(path)

    with DuckDBEngine() as engine:
        orchestrator = Orchestrator(connector, engine, Planner(llm))
        orchestrator.ask("已完成订单的销售额是多少？")
        orchestrator.ask("只看已完成的再算一次。")

    assert connector.read_count == 1
    assert len(orchestrator.history) == 2
    assert orchestrator.history[0].question == "已完成订单的销售额是多少？"
    assert orchestrator.history[0].sql == sql
    assert orchestrator.history[1].question == "只看已完成的再算一次。"

    second_prompt = llm.calls[1][1]["content"]
    assert "最近对话与结果" in second_prompt
    assert "已完成订单的销售额是多少？" in second_prompt
    assert sql in second_prompt


def test_orchestrator_routes_dictionary_questions_without_calling_llm(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    write_orders(path)
    llm = MockLLM(json.dumps({"sql": "SELECT 1 AS ok", "chart": None}))
    qa = DictionaryQA(
        DataDictionaryRetriever(
            [
                DictionaryEntry(
                    id="orders.csv:column:sales",
                    text="字段 sales：销售额，数值型。",
                    source="orders.csv",
                    column="sales",
                )
            ]
        )
    )

    with DuckDBEngine() as engine:
        orchestrator = Orchestrator(CsvConnector(path), engine, Planner(llm), dictionary_qa=qa)
        dictionary_answer = orchestrator.ask("sales 是什么意思？")
        dictionary_trace = orchestrator.last_trace
        analysis_answer = orchestrator.ask("已完成订单的销售额是多少？")

    assert isinstance(dictionary_answer, DictionaryAnswer)
    assert dictionary_answer.citations[0].column == "sales"
    assert dictionary_trace is not None
    assert dictionary_trace.execution_status == "not_run"
    assert [step.name for step in dictionary_trace.steps] == ["路由判定", "字典检索"]
    assert llm.calls[0][1]["content"].endswith("已完成订单的销售额是多少？")
    assert len(llm.calls) == 1
    assert isinstance(analysis_answer, Answer)
    assert len(orchestrator.history) == 1


def test_orchestrator_builds_profile_and_injects_summary(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    write_orders(path)
    llm = MockLLM(json.dumps({"sql": "SELECT 1 AS ok", "chart": None}))

    with DuckDBEngine() as engine:
        orchestrator = Orchestrator(CsvConnector(path), engine, Planner(llm))
        orchestrator.ask("按渠道统计销售额")

    assert orchestrator.profile.row_count == 3
    assert "数据体检" in llm.calls[0][1]["content"]
    assert "sales" in llm.calls[0][1]["content"]


def test_orchestrator_rejects_empty_question(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    write_orders(path)

    with DuckDBEngine() as engine:
        orchestrator = Orchestrator(CsvConnector(path), engine, Planner(MockLLM()))
        with pytest.raises(ValueError, match="问题不能为空"):
            orchestrator.ask("  ")


def test_format_result_table_truncates() -> None:
    frame = pd.DataFrame({"a": range(25)})

    output = format_result_table(frame, max_rows=5)

    assert "… 仅显示前 5 行，共 25 行" in output


def test_format_result_table_handles_empty_frame() -> None:
    assert format_result_table(pd.DataFrame()) == "（查询结果为空）"


def test_orchestrator_returns_clarification_without_planning(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    write_dirty_orders(path)
    llm = MockLLM(json.dumps({"sql": "SELECT 1 AS ok", "chart": None}))

    with DuckDBEngine() as engine:
        orchestrator = Orchestrator(CsvConnector(path), engine, Planner(llm))
        result = orchestrator.ask("统计销售额")

    assert isinstance(result, Clarification)
    assert llm.calls == []
    assert orchestrator.pending_clarification is not None
    assert isinstance(result.trace, Trace)
    assert result.trace.execution_status == "not_run"
    assert result.trace.profile_references
    assert result.trace.steps[-1].status == "clarify"


def test_orchestrator_executes_after_all_clarifications(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    write_dirty_orders(path)
    llm = MockLLM(json.dumps({"sql": "SELECT 1 AS ok", "chart": None}))

    with DuckDBEngine() as engine:
        orchestrator = Orchestrator(CsvConnector(path), engine, Planner(llm))
        result = orchestrator.ask("统计销售额")
        rounds = 0
        while isinstance(result, Clarification):
            rounds += 1
            result = orchestrator.resolve(result.request.options[0].value)

    assert rounds >= 4
    assert isinstance(result, Answer)
    assert orchestrator.pending_clarification is None
    assert llm.calls
    prompt = llm.calls[0][1]["content"]
    assert "用户已确认的数据口径" in prompt
    assert result.sql == "SELECT 1 AS ok"

    assert isinstance(result.trace, Trace)
    assert result.trace.execution_status == "succeeded"
    assert {reference.code for reference in result.trace.profile_references} >= {
        "duplicate_primary_key",
        "missing_values",
        "dirty_numeric_format",
    }
    assert any(
        name == "数值脏格式" for name, _ in result.trace.confirmed_policies
    )
    assert [step.name for step in result.trace.steps].count("按确认口径处理") == rounds


def test_orchestrator_rejects_invalid_clarification_option(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    write_dirty_orders(path)

    with DuckDBEngine() as engine:
        orchestrator = Orchestrator(CsvConnector(path), engine, Planner(MockLLM()))
        orchestrator.ask("统计销售额")

        with pytest.raises(ValueError, match="无效的澄清选项"):
            orchestrator.resolve("not-an-option")


def test_orchestrator_rejects_new_question_while_pending(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    write_dirty_orders(path)

    with DuckDBEngine() as engine:
        orchestrator = Orchestrator(CsvConnector(path), engine, Planner(MockLLM()))
        orchestrator.ask("统计销售额")

        with pytest.raises(RuntimeError, match="未完成的澄清"):
            orchestrator.ask("另一个问题")


def test_orchestrator_resolve_without_pending_raises(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    write_orders(path)

    with DuckDBEngine() as engine:
        orchestrator = Orchestrator(CsvConnector(path), engine, Planner(MockLLM()))

        with pytest.raises(RuntimeError, match="没有待澄清"):
            orchestrator.resolve("dedupe_keep_first")

def test_orchestrator_records_failed_trace_when_repair_exhausted(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    write_orders(path)
    bad_sql = "SELECT missing_column FROM orders"
    llm = SequenceLLM([json.dumps({"sql": bad_sql, "chart": None})] * 3)

    with DuckDBEngine() as engine:
        orchestrator = Orchestrator(CsvConnector(path), engine, Planner(llm))
        with pytest.raises(RepairFailed, match="共尝试 3 次"):
            orchestrator.ask("已完成订单的销售额是多少？")

    trace = orchestrator.last_trace
    assert trace is not None
    assert trace.execution_status == "failed"
    assert trace.attempts == 3
    assert trace.generated_sql == bad_sql
    assert trace.steps[-1].name == "SQL 规划与执行"
    assert trace.steps[-1].status == "failed"
    assert "missing_column" in (trace.steps[-1].detail or "")


def test_orchestrator_renders_chart_from_plan(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    write_orders(path)
    sql = (
        "SELECT channel, SUM(sales) AS total FROM orders "
        "WHERE status = '已完成' GROUP BY channel ORDER BY total DESC"
    )
    chart = {"type": "bar", "x": "channel", "y": "total", "title": "各渠道销售额"}
    llm = MockLLM(json.dumps({"sql": sql, "chart": chart}, ensure_ascii=False))

    with DuckDBEngine() as engine:
        orchestrator = Orchestrator(CsvConnector(path), engine, Planner(llm))
        answer = orchestrator.ask("按渠道统计已完成销售额")

    assert answer.chart == chart
    assert answer.chart_png is not None
    assert answer.chart_png.startswith(b"\x89PNG\r\n\x1a\n")
    chart_step = next(step for step in answer.trace.steps if step.name == "图表")
    assert chart_step.status == "ok"
    assert "bar" in chart_step.summary


def test_orchestrator_skips_invalid_chart_without_failing(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    write_orders(path)
    sql = "SELECT channel, SUM(sales) AS total FROM orders GROUP BY channel"
    chart = {"type": "bar", "x": "missing", "y": "total"}
    llm = MockLLM(json.dumps({"sql": sql, "chart": chart}))

    with DuckDBEngine() as engine:
        orchestrator = Orchestrator(CsvConnector(path), engine, Planner(llm))
        answer = orchestrator.ask("按渠道统计销售额")

    assert answer.chart is None
    assert answer.chart_png is None
    assert int(answer.dataframe["total"].sum()) == 1299
    chart_step = next(step for step in answer.trace.steps if step.name == "图表")
    assert chart_step.status == "skipped"
    assert "不在结果列中" in (chart_step.detail or "")


def test_orchestrator_uses_llm_conclusion_when_configured(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    write_orders(path)
    sql = "SELECT SUM(sales) AS total FROM orders WHERE status = '已完成'"
    planner_llm = MockLLM(json.dumps({"sql": sql, "chart": None}))
    composer_llm = MockLLM("已完成订单的销售额合计为 300。")

    with DuckDBEngine() as engine:
        orchestrator = Orchestrator(
            CsvConnector(path),
            engine,
            Planner(planner_llm),
            conclusion_composer=ConclusionComposer(composer_llm),
        )
        answer = orchestrator.ask("已完成订单的销售额是多少？")

    assert answer.conclusion == "已完成订单的销售额合计为 300。"
    assert answer.conclusion_source == "llm"
    conclusion_step = next(step for step in answer.trace.steps if step.name == "结论生成")
    assert conclusion_step.status == "ok"
    assert conclusion_step.detail == answer.conclusion
    prompt = composer_llm.calls[0][1]["content"]
    assert "已完成订单的销售额是多少？" in prompt
    assert sql in prompt
    assert "300" in prompt


def test_orchestrator_falls_back_to_template_when_conclusion_fails(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    write_orders(path)
    sql = "SELECT SUM(sales) AS total FROM orders WHERE status = '已完成'"
    planner_llm = MockLLM(json.dumps({"sql": sql, "chart": None}))

    with DuckDBEngine() as engine:
        orchestrator = Orchestrator(
            CsvConnector(path),
            engine,
            Planner(planner_llm),
            conclusion_composer=ConclusionComposer(FailingLLM(LLMError("模型超时"))),
        )
        answer = orchestrator.ask("已完成订单的销售额是多少？")

    assert answer.conclusion_source == "template"
    assert answer.conclusion == "total = 300。"
    conclusion_step = next(step for step in answer.trace.steps if step.name == "结论生成")
    assert conclusion_step.status == "fallback"
    assert "模型超时" in (conclusion_step.detail or "")
