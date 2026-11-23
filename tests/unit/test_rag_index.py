import pytest

from prizm.rag.index import BM25Index, tokenize
from prizm.rag.ingest import DictionaryEntry


def entry(identifier: str, text: str) -> DictionaryEntry:
    return DictionaryEntry(id=identifier, text=text, source="dict")


def test_tokenize_splits_chinese_bigrams_and_ascii_words() -> None:
    tokens = tokenize("Sales 销售额 order_id")

    assert "sales" in tokens
    assert "order_id" in tokens
    assert "销售" in tokens
    assert "售额" in tokens


def test_bm25_ranks_matching_column_first() -> None:
    index = BM25Index(
        [
            entry("sales", "字段 sales：销售额，数值型。"),
            entry("channel", "字段 channel：销售渠道，类别型。"),
            entry("region", "字段 region：地区，类别型。"),
        ]
    )

    results = index.search("销售额字段是什么", top_k=2)

    assert results
    assert results[0][0].id == "sales"
    assert results[0][1] > 0


def test_bm25_english_query_matches_ascii_token() -> None:
    index = BM25Index(
        [
            entry("sales", "字段 sales：销售额，数值型。"),
            entry("channel", "字段 channel：销售渠道，类别型。"),
        ]
    )

    results = index.search("channel 含义", top_k=1)

    assert results[0][0].id == "channel"


def test_bm25_empty_index_and_query_return_empty() -> None:
    assert BM25Index([]).search("销售额") == ()
    index = BM25Index([entry("sales", "字段 sales：销售额。")])
    assert index.search("   ") == ()


def test_bm25_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        BM25Index([entry("sales", "sales")]).search("sales", top_k=0)
