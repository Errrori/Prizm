"""数据字典检索与问答：BM25 召回，返回可溯源的原文片段。"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from prizm.rag.index import BM25Index
from prizm.rag.ingest import DictionaryEntry

DEFAULT_TOP_K = 3

_TERM_MARKERS = (
    "什么是",
    "是什么意思",
    "什么意思",
    "含义",
    "代表什么",
    "指的是",
    "定义",
    "字段说明",
    "术语",
    "数据字典",
    "怎么理解",
    "解释一下",
    "取值",
    "枚举",
    "怎么算",
)
_STRONG_TERM_MARKERS = (
    "含义",
    "定义",
    "是什么意思",
    "什么意思",
    "代表什么",
    "指的是",
    "术语",
    "字段说明",
    "数据字典",
)
_ANALYSIS_MARKERS = (
    "统计",
    "计算",
    "汇总",
    "求和",
    "平均",
    "多少",
    "排名",
    "排序",
    "趋势",
    "同比",
    "环比",
    "分布",
    "占比",
    "预测",
    "图表",
    "最高",
    "最低",
    "按渠道",
    "按月",
)


def is_dictionary_question(question: str) -> bool:
    """判断是否走数据字典检索。

    DEBT-016: 路由基于关键词启发式, 分析类问题可能误判或漏判。
    """
    text = question.strip()
    if not text:
        return False
    if not any(marker in text for marker in _TERM_MARKERS):
        return False
    if any(marker in text for marker in _STRONG_TERM_MARKERS):
        return True
    return not any(marker in text for marker in _ANALYSIS_MARKERS)


def _query_mentions_column(query: str, column: str | None) -> bool:
    if not column:
        return False
    candidates = {column, column.replace("_", " ")}
    for candidate in candidates:
        if re.fullmatch(r"[A-Za-z0-9_ ]+", candidate):
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(candidate)}(?![A-Za-z0-9_])"
            if re.search(pattern, query, re.IGNORECASE):
                return True
        elif candidate in query:
            return True
    return False


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    entry: DictionaryEntry
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class Citation:
    index: int
    entry_id: str
    text: str
    source: str
    column: str | None = None


@dataclass(frozen=True, slots=True)
class DictionaryAnswer:
    question: str
    text: str
    citations: tuple[Citation, ...] = ()


class DataDictionaryRetriever:
    def __init__(
        self,
        entries: Sequence[DictionaryEntry],
        *,
        default_top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if default_top_k < 1:
            raise ValueError("default_top_k 必须 >= 1")
        self._entries = tuple(entries)
        self._index = BM25Index(self._entries)
        self._default_top_k = default_top_k

    @property
    def entries(self) -> tuple[DictionaryEntry, ...]:
        return self._entries

    def search(self, query: str, *, top_k: int | None = None) -> tuple[RetrievalHit, ...]:
        limit = self._default_top_k if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k 必须 >= 1")
        if not self._entries:
            return ()

        results = self._index.search(query, top_k=len(self._entries))
        matched_columns = [
            (entry, score)
            for entry, score in results
            if _query_mentions_column(query, entry.column)
        ]
        if matched_columns:
            matched_ids = {entry.id for entry, _ in matched_columns}
            rest = [
                (entry, score)
                for entry, score in results
                if entry.id not in matched_ids
            ]
            selected = (matched_columns + rest)[:limit]
        else:
            selected = list(results)[:limit]
        return tuple(
            RetrievalHit(entry=entry, score=score, rank=rank)
            for rank, (entry, score) in enumerate(selected, 1)
        )


class DictionaryQA:
    """检索式问答：直接返回带出处的字典片段，不让模型脱离原文发挥。"""

    # DEBT-017: 当前为检索片段拼接, 未用 LLM 归纳成自然语言答案。
    def __init__(self, retriever: DataDictionaryRetriever) -> None:
        self._retriever = retriever

    @property
    def retriever(self) -> DataDictionaryRetriever:
        return self._retriever

    def answer(self, question: str) -> DictionaryAnswer:
        if not question.strip():
            raise ValueError("问题不能为空")
        hits = self._retriever.search(question)
        if not hits:
            return DictionaryAnswer(
                question=question,
                text=f"数据字典中未找到与“{question.strip()}”相关的条目，未作猜测。",
                citations=(),
            )

        citations = tuple(
            Citation(
                index=hit.rank,
                entry_id=hit.entry.id,
                text=hit.entry.text,
                source=hit.entry.source,
                column=hit.entry.column,
            )
            for hit in hits
        )
        lines = ["根据数据字典："]
        for citation in citations:
            suffix = f"（来源：{citation.source}"
            if citation.column:
                suffix += f"；字段：{citation.column}"
            suffix += "）"
            lines.append(f"[{citation.index}] {citation.text}{suffix}")
        return DictionaryAnswer(
            question=question,
            text="\n".join(lines),
            citations=citations,
        )
