import sqlite3
from pathlib import Path

import pytest

from prizm.datasource.sqlite_connector import SqliteConnector, list_sqlite_tables
from prizm.engine.duckdb_engine import DuckDBEngine


def _write_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, 渠道 TEXT, 销售额 INTEGER);
        INSERT INTO orders (渠道, 销售额) VALUES ('线上', 100), ('线下', 200);
        CREATE TABLE traffic (城市 TEXT, pv INTEGER);
        INSERT INTO traffic (城市, pv) VALUES ('北京', 10);
        """
    )
    conn.commit()
    conn.close()


def test_list_sqlite_tables_excludes_internal(tmp_path: Path) -> None:
    path = tmp_path / "shop.sqlite"
    _write_db(path)

    assert list_sqlite_tables(path) == ("orders", "traffic")


def test_connector_reads_table_and_registers(tmp_path: Path) -> None:
    path = tmp_path / "shop.sqlite"
    _write_db(path)
    connector = SqliteConnector(path, table="orders")

    frame = connector.load()

    assert connector.view_name == "orders"
    assert connector.table == "orders"
    assert frame["渠道"].tolist() == ["线上", "线下"]
    assert connector.load() is frame

    with DuckDBEngine() as engine:
        connector.register(engine)
        result = engine.execute("SELECT SUM(销售额) AS total FROM orders")

    assert int(result.loc[0, "total"]) == 300


def test_connector_rejects_unknown_table(tmp_path: Path) -> None:
    path = tmp_path / "shop.sqlite"
    _write_db(path)

    with pytest.raises(ValueError, match="数据表不存在: missing"):
        SqliteConnector(path, table="missing").load()


def test_connector_rejects_empty_database(tmp_path: Path) -> None:
    path = tmp_path / "empty.sqlite"
    sqlite3.connect(path).close()

    assert list_sqlite_tables(path) == ()
    with pytest.raises(ValueError, match="没有可读取的数据表"):
        SqliteConnector(path, table="orders").load()


def test_list_tables_rejects_non_sqlite_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.db"
    path.write_bytes(b"not a database")

    with pytest.raises(ValueError, match="无法读取 SQLite 文件"):
        list_sqlite_tables(path)


def test_connector_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="数据文件不存在"):
        SqliteConnector(tmp_path / "missing.db", table="orders")
