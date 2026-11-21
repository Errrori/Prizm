from types import SimpleNamespace
from typing import Any

import pytest

from prizm.config import Settings
from prizm.llm.adapter import LiteLLMAdapter, LLMConfigError, LLMError, MockLLM, build_llm


def make_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_mock_llm_returns_response_and_records_calls() -> None:
    llm = MockLLM('{"sql": "SELECT 1", "chart": null}')

    text = llm.complete([{"role": "user", "content": "hi"}])

    assert text == '{"sql": "SELECT 1", "chart": null}'
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

    assert adapter.complete([{"role": "user", "content": "hi"}], temperature=0.2) == "ok"
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
