import json
from pathlib import Path

import pytest

from prizm.agent.orchestrator import Orchestrator
from prizm.agent.planner import Planner
from prizm.datasource.csv_connector import CsvConnector
from prizm.engine.duckdb_engine import DuckDBEngine
from prizm.llm.adapter import MockLLM
from prizm.profiling.profiler import profile_frame

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


@pytest.mark.integration
def test_dirty_orders_profile_detects_all_seeded_anomalies() -> None:
    dataset = REPO_ROOT / "eval" / "datasets" / "orders_dirty.csv"

    report = profile_frame(CsvConnector(dataset).load())

    assert report.anomaly_codes >= {
        "missing_values",
        "exact_duplicate_rows",
        "duplicate_primary_key",
        "dirty_numeric_format",
    }
    assert report.duplicates.exact_duplicate_rows == 1
    assert report.duplicates.key_columns == ("order_id",)
    assert report.duplicates.key_duplicate_group_count == 2
    assert {
        anomaly.column for anomaly in report.anomalies if anomaly.code == "missing_values"
    } == {"quantity", "status"}
    assert [
        (anomaly.column, anomaly.count)
        for anomaly in report.anomalies
        if anomaly.code == "dirty_numeric_format"
    ] == [("sales", 1)]
