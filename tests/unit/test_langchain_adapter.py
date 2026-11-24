"""LangChain 适配层单测：不联网，注入假 ChatModel。"""

from __future__ import annotations

import pytest

from prizm.config import Settings
from prizm.llm.adapter import LLMConfigError, LLMError, MockLLM
from prizm.llm.langchain_adapter import (
    LangChainLLMAdapter,
    build_langchain_llm,
    extract_text,
    extract_usage,
)


class _Message:
    def __init__(self, content, usage=None):
        self.content = content
        self.usage_metadata = usage or {}


class _FakeChat:
    def __init__(self, replies=None, chunks=None):
        self._replies = list(replies or [])
        self._chunks = list(chunks or [])
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self._replies.pop(0)

    def stream(self, messages):
        self.calls.append(messages)
        yield from self._chunks


class _Ledger:
    def __init__(self) -> None:
        self.records = []
        self.budget: int | None = None

    def ensure_available(self, budget: int) -> None:
        self.budget = budget

    def record(self, usage) -> None:
        self.records.append(usage)


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_complete_maps_usage_and_converts_messages() -> None:
    chat = _FakeChat(
        replies=[_Message("SELECT 1", {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5})]
    )
    ledger = _Ledger()
    adapter = LangChainLLMAdapter(chat, ledger=ledger, daily_token_budget=100)

    completion = adapter.complete([{"role": "user", "content": "看下销售额"}])

    assert completion.text == "SELECT 1"
    assert completion.usage.prompt_tokens == 3
    assert completion.usage.completion_tokens == 2
    assert completion.usage.total_tokens == 5
    assert ledger.budget == 100
    assert ledger.records[0].total_tokens == 5

    sent = chat.calls[0][0]
    assert sent.type == "human"
    assert sent.content == "看下销售额"


def test_complete_empty_content_raises() -> None:
    adapter = LangChainLLMAdapter(_FakeChat(replies=[_Message("   ")]))
    with pytest.raises(LLMError):
        adapter.complete([{"role": "user", "content": "hi"}])


def test_stream_yields_chunks_and_records_usage() -> None:
    chat = _FakeChat(
        chunks=[_Message("SELECT "), _Message("1", {"prompt_tokens": 1, "total_tokens": 7})]
    )
    ledger = _Ledger()
    adapter = LangChainLLMAdapter(chat, ledger=ledger, daily_token_budget=50)

    pieces = list(adapter.stream([{"role": "user", "content": "hi"}]))

    assert pieces == ["SELECT ", "1"]
    assert ledger.records[-1].total_tokens == 7


def test_stream_empty_raises() -> None:
    adapter = LangChainLLMAdapter(_FakeChat(chunks=[]))
    with pytest.raises(LLMError):
        list(adapter.stream([{"role": "user", "content": "hi"}]))


def test_extract_handles_content_blocks_and_missing_usage() -> None:
    message = _Message([{"type": "text", "text": "ab"}, "c"])
    assert extract_text(message) == "abc"
    assert extract_usage(message).total_tokens == 0


def test_build_langchain_mock_returns_mock_llm() -> None:
    adapter = build_langchain_llm(_settings(default_model="mock"))
    assert isinstance(adapter, MockLLM)


def test_build_langchain_with_injected_chat() -> None:
    adapter = build_langchain_llm(
        _settings(default_model="gpt-4o-mini", openai_api_key="sk-test"),
        chat=_FakeChat(replies=[_Message("ok")]),
    )
    assert isinstance(adapter, LangChainLLMAdapter)


def test_build_langchain_missing_key_raises() -> None:
    settings = _settings(default_model="gpt-4o-mini", openai_api_key=None)
    with pytest.raises(LLMConfigError):
        build_langchain_llm(settings)
