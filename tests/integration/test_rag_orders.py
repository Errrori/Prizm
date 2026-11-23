from pathlib import Path

import pytest

from prizm.datasource.csv_connector import CsvConnector
from prizm.profiling.profiler import profile_frame
from prizm.rag.ingest import build_data_dictionary
from prizm.rag.retriever import DataDictionaryRetriever, DictionaryQA

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_orders_dictionary_cites_sales_and_channel() -> None:
    frame = CsvConnector(REPO_ROOT / "eval" / "datasets" / "orders.csv").load()
    entries = build_data_dictionary(
        frame,
        profile=profile_frame(frame),
        source="orders.csv",
    )
    qa = DictionaryQA(DataDictionaryRetriever(entries))

    sales = qa.answer("sales 字段是什么意思？")
    channel = qa.answer("渠道字段的含义是什么？")

    assert sales.citations[0].column == "sales"
    assert "数值范围" in sales.text
    assert channel.citations[0].column == "channel"
    assert "来源：orders.csv" in channel.text
