import json
from pathlib import Path

import pandas as pd
import pytest

from prizm.agent.orchestrator import Orchestrator, format_result_table
from prizm.agent.planner import Planner
from prizm.datasource.csv_connector import CsvConnector
from prizm.engine.duckdb_engine import DuckDBEngine
from prizm.llm.adapter import MockLLM


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


def test_orchestrator_runs_plan_end_to_end(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    write_orders(path)
    connector = CountingConnector(path)
    llm = MockLLM(
        json.dumps(
            {
                "sql": (
                    "SELECT SUM(sales) AS total FROM orders "
                    "WHERE status = '已完成'"
                ),
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
