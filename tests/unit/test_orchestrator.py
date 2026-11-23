import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest

from prizm.agent.orchestrator import (
    Answer,
    Clarification,
    Orchestrator,
    format_result_table,
)
from prizm.agent.planner import Planner
from prizm.datasource.csv_connector import CsvConnector
from prizm.engine.duckdb_engine import DuckDBEngine
from prizm.llm.adapter import Completion, Message, MockLLM
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
        analysis_answer = orchestrator.ask("已完成订单的销售额是多少？")

    assert isinstance(dictionary_answer, DictionaryAnswer)
    assert dictionary_answer.citations[0].column == "sales"
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
