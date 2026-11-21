from types import SimpleNamespace
from typing import Any

import pytest

from prizm.config import Settings
from prizm.llm.adapter import (
    LiteLLMAdapter,
    LLMConfigError,
    LLMError,
    MockLLM,
    build_llm,
    extract_usage,
)
from prizm.utils.tokens import DailyUsageLedger, QuotaExceededError, Usage


def make_response(
    content: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cost: float | None = None,
) -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    hidden = {} if cost is None else {"response_cost": cost}
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage,
        _hidden_params=hidden,
    )


def test_mock_llm_returns_response_and_records_calls() -> None:
    llm = MockLLM('{"sql": "SELECT 1", "chart": null}')

    completion = llm.complete([{"role": "user", "content": "hi"}])

    assert completion.text == '{"sql": "SELECT 1", "chart": null}'
    assert completion.usage.total_tokens == 0
    assert llm.calls == [[{"role": "user", "content": "hi"}]]


def test_litellm_adapter_passes_arguments() -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return make_response("ok")

    adapter = LiteLLMAdapter(
        "gpt-test",
        api_key="secret",
        api_base="https://proxy.example/v1",
        completion_fn=fake_completion,
    )

    assert adapter.complete([{"role": "user", "content": "hi"}], temperature=0.2).text == "ok"
    assert captured["model"] == "gpt-test"
    assert captured["temperature"] == 0.2
    assert captured["api_key"] == "secret"
    assert captured["api_base"] == "https://proxy.example/v1"
    assert captured["messages"] == [{"role": "user", "content": "hi"}]


def test_litellm_adapter_omits_api_base_when_unset() -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return make_response("ok")

    adapter = LiteLLMAdapter("gpt-test", api_key="secret", completion_fn=fake_completion)

    adapter.complete([{"role": "user", "content": "hi"}])

    assert "api_base" not in captured


def test_litellm_adapter_wraps_call_errors() -> None:
    def failing_completion(**kwargs: Any) -> SimpleNamespace:
        raise RuntimeError("network down")

    adapter = LiteLLMAdapter("gpt-test", completion_fn=failing_completion)

    with pytest.raises(LLMError, match="network down"):
        adapter.complete([{"role": "user", "content": "hi"}])


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="   "))]),
    ],
)
def test_litellm_adapter_rejects_invalid_response(response: SimpleNamespace) -> None:
    adapter = LiteLLMAdapter("gpt-test", completion_fn=lambda **_: response)

    with pytest.raises(LLMError, match="LLM 返回"):
        adapter.complete([{"role": "user", "content": "hi"}])


def test_litellm_adapter_returns_usage_and_cost() -> None:
    response = make_response(
        "ok",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cost=0.002,
    )
    adapter = LiteLLMAdapter("gpt-test", completion_fn=lambda **_: response)

    completion = adapter.complete([{"role": "user", "content": "hi"}])

    assert completion.text == "ok"
    assert completion.usage == Usage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cost_usd=0.002,
    )


def test_litellm_adapter_records_usage_in_ledger() -> None:
    ledger = DailyUsageLedger()
    response = make_response("ok", prompt_tokens=10, completion_tokens=5)
    adapter = LiteLLMAdapter(
        "gpt-test",
        completion_fn=lambda **_: response,
        ledger=ledger,
        daily_token_budget=100,
    )

    adapter.complete([{"role": "user", "content": "hi"}])

    assert ledger.snapshot().total_tokens == 15


def test_litellm_adapter_blocks_when_budget_exhausted() -> None:
    calls: list[dict[str, Any]] = []
    ledger = DailyUsageLedger()
    ledger.record(Usage(total_tokens=100))

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return make_response("ok")

    adapter = LiteLLMAdapter(
        "gpt-test",
        completion_fn=fake_completion,
        ledger=ledger,
        daily_token_budget=100,
    )

    with pytest.raises(QuotaExceededError, match="额度已用完"):
        adapter.complete([{"role": "user", "content": "hi"}])
    assert calls == []


def test_extract_usage_handles_missing_fields() -> None:
    assert extract_usage(SimpleNamespace()) == Usage()
    assert extract_usage({"usage": {"prompt_tokens": 2, "total_tokens": 2}}) == Usage(
        prompt_tokens=2,
        total_tokens=2,
    )


def test_build_llm_uses_mock_by_default() -> None:
    llm = build_llm(Settings(_env_file=None))

    assert isinstance(llm, MockLLM)


def test_build_llm_uses_litellm_when_model_configured() -> None:
    settings = Settings(
        _env_file=None,
        default_model="gpt-test",
        openai_api_key="secret",
    )

    llm = build_llm(settings)

    assert isinstance(llm, LiteLLMAdapter)
    assert llm.model == "gpt-test"
    assert llm.api_key == "secret"


def test_build_llm_passes_api_base() -> None:
    settings = Settings(
        _env_file=None,
        default_model="gpt-test",
        openai_api_key="secret",
        llm_api_base="https://proxy.example/v1",
    )

    llm = build_llm(settings)

    assert isinstance(llm, LiteLLMAdapter)
    assert llm.api_base == "https://proxy.example/v1"


def test_build_llm_passes_ledger_and_budget() -> None:
    ledger = DailyUsageLedger()
    settings = Settings(
        _env_file=None,
        default_model="gpt-test",
        openai_api_key="secret",
        daily_token_budget=123,
    )

    llm = build_llm(settings, ledger=ledger)

    assert isinstance(llm, LiteLLMAdapter)
    assert llm.ledger is ledger
    assert llm.daily_token_budget == 123


def test_build_llm_requires_provider_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(_env_file=None, default_model="gpt-test")

    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        build_llm(settings)


def test_build_llm_rejects_mismatched_provider_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    settings = Settings(
        _env_file=None,
        default_model="qwen-plus",
        openai_api_key="secret",
    )

    with pytest.raises(LLMConfigError, match="DASHSCOPE_API_KEY"):
        build_llm(settings)
