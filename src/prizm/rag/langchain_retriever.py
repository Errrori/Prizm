"""把自研 BM25 索引包装成 LangChain ``BaseRetriever``，供 LCEL/RAG 链路复用。"""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from prizm.rag.index import BM25Index
from prizm.rag.ingest import DictionaryEntry


class BM25LangChainRetriever(BaseRetriever):
    """LangChain 检索器接口 → 项目内 BM25Index，保留来源/字段元数据。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: BM25Index
    top_k: int = Field(default=3, ge=1)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        hits = self.index.search(query, top_k=self.top_k)
        documents: list[Document] = []
        for rank, (entry, score) in enumerate(hits, start=1):
            documents.append(
                Document(
                    page_content=entry.text,
                    metadata={
                        "id": entry.id,
                        "source": entry.source,
                        "column": entry.column,
                        "score": float(score),
                        "rank": rank,
                    },
                )
            )
        return documents


def build_langchain_retriever(
    entries: list[DictionaryEntry] | tuple[DictionaryEntry, ...],
    *,
    top_k: int = 3,
) -> BM25LangChainRetriever:
    return BM25LangChainRetriever(index=BM25Index(entries), top_k=top_k)


__all__ = ["BM25LangChainRetriever", "build_langchain_retriever"]
