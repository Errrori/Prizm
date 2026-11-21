import json
from pathlib import Path

import pytest

from prizm.agent.orchestrator import Orchestrator
from prizm.agent.planner import Planner
from prizm.datasource.csv_connector import CsvConnector
from prizm.engine.duckdb_engine import DuckDBEngine
from prizm.llm.adapter import MockLLM

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_orders_q1_completed_sales_via_minimal_chain() -> None:
    dataset = REPO_ROOT / "eval" / "datasets" / "orders.csv"
    sql = (
        "SELECT SUM(sales) AS total_sales FROM orders "
        "WHERE status = '已完成' "
        "AND order_date >= '2026-01-01' AND order_date < '2026-04-01'"
    )
    llm = MockLLM(json.dumps({"sql": sql, "chart": None}))

    with DuckDBEngine() as engine:
        orchestrator = Orchestrator(CsvConnector(dataset), engine, Planner(llm))
        answer = orchestrator.ask("统计 2026 年第一季度状态为已完成的订单销售额总额。")

    assert int(answer.dataframe.loc[0, "total_sales"]) == 11200
