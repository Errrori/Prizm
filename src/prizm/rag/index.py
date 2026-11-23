"""BM25 索引：中文按二元字组切分，英文/数字按词切分。"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rank_bm25 import BM25Plus

from prizm.rag.ingest import DictionaryEntry

_ASCII_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """轻量中英文分词，避免为中文检索引入额外分词依赖。"""
    lowered = text.lower()
    tokens = [match.group(0) for match in _ASCII_TOKEN_RE.finditer(lowered)]
    for run in _CJK_RUN_RE.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


class BM25Index:
    def __init__(self, entries: Sequence[DictionaryEntry]) -> None:
        self.entries = tuple(entries)
        corpus = [tokenize(entry.text) for entry in self.entries]
        self._bm25: BM25Plus | None = BM25Plus(corpus) if corpus else None

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> tuple[tuple[DictionaryEntry, float], ...]:
        if top_k < 1:
            raise ValueError("top_k 必须 >= 1")
        if self._bm25 is None:
            return ()
        query_tokens = tokenize(query)
        if not query_tokens:
            return ()

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )
        results: list[tuple[DictionaryEntry, float]] = []
        for index, score in ranked:
            if score <= 0:
                continue
            results.append((self.entries[index], float(score)))
            if len(results) >= top_k:
                break
        return tuple(results)
