from pathlib import Path

import pandas as pd
import pytest

from prizm.datasource.base import sanitize_view_name
from prizm.datasource.csv_connector import CsvConnector
from prizm.engine.duckdb_engine import DuckDBEngine


def test_sanitize_view_name() -> None:
    assert sanitize_view_name("orders.csv") == "orders_csv"
    assert sanitize_view_name("订单 明细") == "dataset"
    assert sanitize_view_name("2026-sales") == "t_2026_sales"
    assert sanitize_view_name("  ") == "dataset"


def test_connector_reads_utf8_and_registers(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    pd.DataFrame({"渠道": ["线上", "线下"], "销售额": [100, 200]}).to_csv(
        path, index=False, encoding="utf-8-sig"
    )
    connector = CsvConnector(path)

    assert connector.view_name == "orders"
    assert connector.load() is connector.load()

    with DuckDBEngine() as engine:
        connector.register(engine)
        result = engine.execute("SELECT SUM(销售额) AS total FROM orders")

    assert int(result.loc[0, "total"]) == 300


def test_connector_falls_back_to_gbk(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    pd.DataFrame({"渠道": ["线上"], "销售额": [100]}).to_csv(
        path, index=False, encoding="gbk"
    )

    frame = CsvConnector(path).load()

    assert frame["渠道"].tolist() == ["线上"]


def test_connector_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="数据文件不存在"):
        CsvConnector(tmp_path / "missing.csv")
