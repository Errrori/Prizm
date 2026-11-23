from pathlib import Path

import pandas as pd
import pytest

from prizm.datasource.excel_connector import ExcelConnector, list_excel_sheets
from prizm.engine.duckdb_engine import DuckDBEngine


def _write_workbook(path: Path) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"渠道": ["线上", "线下"], "销售额": [100, 200]}).to_excel(
            writer, sheet_name="订单", index=False
        )
        pd.DataFrame({"城市": ["北京"], "流量": [10]}).to_excel(
            writer, sheet_name="流量", index=False
        )


def test_list_excel_sheets(tmp_path: Path) -> None:
    path = tmp_path / "sales.xlsx"
    _write_workbook(path)

    assert list_excel_sheets(path) == ("订单", "流量")


def test_connector_defaults_to_first_sheet(tmp_path: Path) -> None:
    path = tmp_path / "sales.xlsx"
    _write_workbook(path)
    connector = ExcelConnector(path)

    frame = connector.load()

    assert connector.view_name == "sales"
    assert frame["渠道"].tolist() == ["线上", "线下"]
    assert connector.load() is frame


def test_connector_reads_named_sheet_and_registers(tmp_path: Path) -> None:
    path = tmp_path / "sales.xlsx"
    _write_workbook(path)
    connector = ExcelConnector(path, view_name="traffic", sheet_name="流量")

    with DuckDBEngine() as engine:
        connector.register(engine)
        result = engine.execute("SELECT SUM(流量) AS total FROM traffic")

    assert connector.sheet_name == "流量"
    assert int(result.loc[0, "total"]) == 10


def test_connector_rejects_unknown_sheet(tmp_path: Path) -> None:
    path = tmp_path / "sales.xlsx"
    _write_workbook(path)

    with pytest.raises(ValueError, match="工作表不存在: 缺失"):
        ExcelConnector(path, sheet_name="缺失").load()


def test_connector_rejects_legacy_xls(tmp_path: Path) -> None:
    path = tmp_path / "sales.xls"
    path.write_bytes(b"not really xls")

    with pytest.raises(ValueError, match=r"旧版 \.xls"):
        ExcelConnector(path)


def test_connector_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="数据文件不存在"):
        ExcelConnector(tmp_path / "missing.xlsx")
