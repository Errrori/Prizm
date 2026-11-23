from pathlib import Path

import pandas as pd
import pytest

from prizm.profiling.profiler import profile_frame
from prizm.rag.ingest import (
    DictionaryEntry,
    build_data_dictionary,
    load_dictionary_file,
    parse_dictionary_text,
)


def test_entry_rejects_empty_fields() -> None:
    with pytest.raises(ValueError, match="id"):
        DictionaryEntry(id=" ", text="text", source="source")
    with pytest.raises(ValueError, match="text"):
        DictionaryEntry(id="id", text=" ", source="source")
    with pytest.raises(ValueError, match="source"):
        DictionaryEntry(id="id", text="text", source=" ")
    with pytest.raises(ValueError, match="column"):
        DictionaryEntry(id="id", text="text", source="source", column=" ")


def test_build_data_dictionary_generates_overview_and_columns() -> None:
    frame = pd.DataFrame(
        {
            "order_id": [1, 2],
            "channel": ["线上", "线下"],
            "sales": [100, 200],
        }
    )

    entries = build_data_dictionary(frame, source="orders.csv")

    assert len(entries) == 4
    assert entries[0].id == "orders.csv:overview"
    assert "2 行 × 3 列" in entries[0].text
    assert "order_id" in entries[0].text

    sales = next(entry for entry in entries if entry.column == "sales")
    assert "字段 sales" in sales.text
    assert "示例值" in sales.text
    assert "100" in sales.text
    assert "数值范围" in sales.text


def test_build_data_dictionary_can_include_profile_anomalies() -> None:
    frame = pd.DataFrame(
        {
            "order_id": [1, 1],
            "quantity": [1, None],
            "sales": ["1,200", "800"],
        }
    )
    profile = profile_frame(frame)

    entries = build_data_dictionary(frame, profile=profile, source="dirty.csv")

    sales = next(entry for entry in entries if entry.column == "sales")
    quantity = next(entry for entry in entries if entry.column == "quantity")
    assert "异常提示" in sales.text
    assert "缺失" in quantity.text
    assert "体检异常" in entries[0].text


def test_parse_dictionary_text_splits_sections_and_keeps_source() -> None:
    text = "# 字段说明\n\nsales 表示销售额。\n\n# 渠道\n\nchannel 表示销售渠道。"

    entries = parse_dictionary_text(text, source="data_dict.md")

    assert len(entries) == 2
    assert entries[0].id == "data_dict.md#1"
    assert entries[0].source == "data_dict.md"
    assert "字段说明" in entries[0].text
    assert "sales" in entries[0].text
    assert "渠道" in entries[1].text


def test_parse_dictionary_text_splits_long_blocks() -> None:
    text = "\n".join(f"第 {index} 行说明" for index in range(100))

    entries = parse_dictionary_text(text, source="long.md", max_chars=50)

    assert len(entries) > 1
    assert all(len(entry.text) <= 50 for entry in entries)


def test_parse_dictionary_text_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="source"):
        parse_dictionary_text("text", source=" ")
    with pytest.raises(ValueError, match="max_chars"):
        parse_dictionary_text("text", source="source", max_chars=0)


def test_load_dictionary_file_reads_utf8(tmp_path: Path) -> None:
    path = tmp_path / "dict.md"
    path.write_text("# 销售额\n\nsales 表示销售额。", encoding="utf-8")

    entries = load_dictionary_file(path)

    assert entries[0].source == "dict.md"
    assert "sales" in entries[0].text


def test_load_dictionary_file_reads_gbk(tmp_path: Path) -> None:
    path = tmp_path / "dict.txt"
    path.write_text("销售额 表示订单金额。", encoding="gbk")

    entries = load_dictionary_file(path)

    assert "销售额" in entries[0].text


def test_load_dictionary_file_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dictionary_file(tmp_path / "missing.md")
