import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
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


def test_build_llm_passes_retries() -> None:
    settings = Settings(
        _env_file=None,
        default_model="gpt-test",
        openai_api_key="secret",
        llm_max_retries=4,
    )

    llm = build_llm(settings)

    assert isinstance(llm, LiteLLMAdapter)
    assert llm.max_retries == 4


def test_litellm_adapter_passes_retries() -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return make_response("ok")

    adapter = LiteLLMAdapter(
        "gpt-test",
        completion_fn=fake_completion,
        max_retries=3,
    )

    adapter.complete([{"role": "user", "content": "hi"}])

    assert captured["num_retries"] == 3


def test_litellm_adapter_rejects_negative_retries() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        LiteLLMAdapter("gpt-test", max_retries=-1)


def test_litellm_import_does_not_load_dotenv_from_cwd(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("PRIZM_LEAK_TEST=leaked\n", encoding="utf-8")
    env = os.environ.copy()
    env.pop("LITELLM_MODE", None)
    env.pop("PRIZM_LEAK_TEST", None)
    src = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH")]))
    code = (
        "import os\n"
        "from prizm.llm.adapter import LiteLLMAdapter\n"
        "LiteLLMAdapter('gpt-test')._completion()\n"
        "print(os.environ.get('PRIZM_LEAK_TEST', 'CLEAN'))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "CLEAN"


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


def make_stream_chunk(
    content: str | None,
    *,
    usage: tuple[int, int, int] | None = None,
    cost: float | None = None,
) -> SimpleNamespace:
    choices = (
        []
        if content is None
        else [SimpleNamespace(delta=SimpleNamespace(content=content))]
    )
    prompt, completion, total = usage or (0, 0, 0)
    hidden = {} if cost is None else {"response_cost": cost}
    return SimpleNamespace(
        choices=choices,
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        ),
        _hidden_params=hidden,
    )


def test_mock_llm_streams_in_pieces() -> None:
    llm = MockLLM("销售额合计 300。")

    chunks = list(llm.stream([{"role": "user", "content": "hi"}]))

    assert "".join(chunks) == "销售额合计 300。"
    assert len(chunks) > 1
    assert llm.calls == [[{"role": "user", "content": "hi"}]]


def test_litellm_adapter_streams_and_records_usage() -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> Iterator[SimpleNamespace]:
        captured.update(kwargs)
        yield make_stream_chunk("销售")
        yield make_stream_chunk("额")
        yield make_stream_chunk(" 300。")
        yield make_stream_chunk(None, usage=(10, 5, 15), cost=0.002)

    ledger = DailyUsageLedger()
    adapter = LiteLLMAdapter(
        "gpt-test",
        api_key="k",
        completion_fn=fake_completion,
        ledger=ledger,
        daily_token_budget=1000,
    )

    text = "".join(adapter.stream([{"role": "user", "content": "hi"}]))

    assert text == "销售额 300。"
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert ledger.snapshot().total_tokens == 15
    assert ledger.snapshot().cost_usd == pytest.approx(0.002)


def test_litellm_adapter_stream_raises_on_empty() -> None:
    def fake_completion(**kwargs: Any) -> Iterator[SimpleNamespace]:
        yield make_stream_chunk(None)

    adapter = LiteLLMAdapter("gpt-test", api_key="k", completion_fn=fake_completion)

    with pytest.raises(LLMError, match="为空"):
        list(adapter.stream([{"role": "user", "content": "hi"}]))


def test_litellm_adapter_wraps_stream_errors() -> None:
    def fake_completion(**kwargs: Any) -> Iterator[SimpleNamespace]:
        raise RuntimeError("boom")
        yield  # pragma: no cover

    adapter = LiteLLMAdapter("gpt-test", api_key="k", completion_fn=fake_completion)

    with pytest.raises(LLMError, match="流式调用失败"):
        list(adapter.stream([{"role": "user", "content": "hi"}]))
