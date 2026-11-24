"""LangChain BaseRetriever 封装单测：不触网。"""

from __future__ import annotations

import pytest

from prizm.rag.index import BM25Index
from prizm.rag.ingest import DictionaryEntry
from prizm.rag.langchain_retriever import BM25LangChainRetriever, build_langchain_retriever

_ENTRIES = [
    DictionaryEntry(
        id="orders.csv:column:sales",
        text="字段 sales：销售额，数值型。",
        source="orders.csv",
        column="sales",
    ),
    DictionaryEntry(
        id="orders.csv:column:channel",
        text="字段 channel：销售渠道，文本型。",
        source="orders.csv",
        column="channel",
    ),
]


def test_retriever_returns_documents_with_metadata() -> None:
    retriever = build_langchain_retriever(_ENTRIES, top_k=2)

    documents = retriever.invoke("sales 是什么意思")

    assert documents
    assert documents[0].metadata["column"] == "sales"
    assert documents[0].metadata["rank"] == 1
    assert documents[0].page_content.startswith("字段 sales")


def test_retriever_reuses_bm25_index_and_validates_top_k() -> None:
    retriever = BM25LangChainRetriever(index=BM25Index(_ENTRIES), top_k=1)
    assert retriever.invoke("channel")[0].metadata["column"] == "channel"

    with pytest.raises(ValueError):
        BM25LangChainRetriever(index=BM25Index(_ENTRIES), top_k=0)
