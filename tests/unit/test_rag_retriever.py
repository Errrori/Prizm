import pytest

from prizm.rag.ingest import DictionaryEntry
from prizm.rag.retriever import (
    DataDictionaryRetriever,
    DictionaryQA,
    is_dictionary_question,
)


def entry(identifier: str, text: str, *, column: str | None = None) -> DictionaryEntry:
    return DictionaryEntry(
        id=identifier,
        text=text,
        source="orders.csv",
        column=column,
    )


def retriever() -> DataDictionaryRetriever:
    return DataDictionaryRetriever(
        [
            entry("orders.csv:column:sales", "字段 sales：销售额，数值型。", column="sales"),
            entry(
                "orders.csv:column:channel",
                "字段 channel：销售渠道，类别型。",
                column="channel",
            ),
            entry("orders.csv:overview", "数据集 orders.csv：24 行 × 9 列。"),
        ]
    )


def test_search_prefers_explicit_column_over_overview() -> None:
    retriever_with_overview = DataDictionaryRetriever(
        [
            entry(
                "orders.csv:overview",
                "数据集 orders.csv：列 sales, channel, region, product。",
            ),
            entry("orders.csv:column:sales", "字段 sales：销售额。", column="sales"),
            entry("orders.csv:column:channel", "字段 channel：渠道。", column="channel"),
        ]
    )

    hits = retriever_with_overview.search("sales 是什么意思？", top_k=1)

    assert hits[0].entry.column == "sales"


def test_search_returns_ranked_hits() -> None:
    hits = retriever().search("sales 字段是什么意思", top_k=2)

    assert len(hits) == 2
    assert hits[0].rank == 1
    assert hits[0].entry.column == "sales"
    assert hits[0].score > 0


def test_search_respects_top_k_and_rejects_invalid_values() -> None:
    assert len(retriever().search("sales", top_k=1)) == 1
    with pytest.raises(ValueError, match="top_k"):
        retriever().search("sales", top_k=0)
    with pytest.raises(ValueError, match="default_top_k"):
        DataDictionaryRetriever([], default_top_k=0)


def test_dictionary_answer_includes_citations_and_source() -> None:
    answer = DictionaryQA(retriever()).answer("销售额字段是什么意思？")

    assert answer.citations
    assert answer.citations[0].column == "sales"
    assert "sales" in answer.text
    assert "[1]" in answer.text
    assert "来源：orders.csv" in answer.text


def test_dictionary_answer_returns_not_found_without_guessing() -> None:
    answer = DictionaryQA(retriever()).answer("完全不存在的术语是什么？")

    assert answer.citations == ()
    assert "未找到" in answer.text
    assert "未作猜测" in answer.text


def test_dictionary_answer_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="问题不能为空"):
        DictionaryQA(retriever()).answer("   ")


@pytest.mark.parametrize(
    "question",
    [
        "sales 是什么意思？",
        "sales 字段的含义是什么？",
        "解释一下 channel 字段。",
        "order_id 代表什么？",
        "数据字典里有哪些字段？",
        "毛利率怎么算？",
    ],
)
def test_is_dictionary_question_matches_term_questions(question: str) -> None:
    assert is_dictionary_question(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "统计各渠道销售额",
        "按月看销售额趋势",
        "销售额是多少？",
        "解释一下销售额趋势",
        "只看已完成的再算一次",
    ],
)
def test_is_dictionary_question_rejects_analysis_questions(question: str) -> None:
    assert is_dictionary_question(question) is False
