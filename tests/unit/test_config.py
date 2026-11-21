import pytest

from prizm.config import (
    MOCK_MODEL,
    PROVIDER_ANTHROPIC,
    PROVIDER_DASHSCOPE,
    PROVIDER_OPENAI,
    Settings,
    detect_provider,
)


def test_settings_default_to_mock_and_limits() -> None:
    settings = Settings(_env_file=None)

    assert settings.default_model == MOCK_MODEL
    assert settings.is_mock is True
    assert settings.duckdb_memory_limit == "1GB"
    assert settings.duckdb_threads == 2
    assert settings.sql_timeout_seconds == 30.0
    assert settings.openai_api_key is None
    assert settings.llm_api_base is None


def test_settings_read_prefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIZM_DEFAULT_MODEL", "qwen-plus")
    monkeypatch.setenv("PRIZM_DUCKDB_THREADS", "4")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert settings.default_model == "qwen-plus"
    assert settings.is_mock is False
    assert settings.duckdb_threads == 4
    assert settings.dashscope_api_key == "test-key"


def test_settings_reads_llm_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIZM_LLM_API_BASE", "https://proxy.example/v1")

    settings = Settings(_env_file=None)

    assert settings.llm_api_base == "https://proxy.example/v1"


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("qwen-plus", PROVIDER_DASHSCOPE),
        ("dashscope/qwen-max", PROVIDER_DASHSCOPE),
        ("claude-3-5-sonnet-latest", PROVIDER_ANTHROPIC),
        ("anthropic/claude-3-5-sonnet", PROVIDER_ANTHROPIC),
        ("gpt-4o", PROVIDER_OPENAI),
        ("openai/gpt-4o-mini", PROVIDER_OPENAI),
    ],
)
def test_detect_provider_by_model(model: str, provider: str) -> None:
    assert detect_provider(model) == provider


def test_llm_api_key_only_uses_current_provider() -> None:
    settings = Settings(
        _env_file=None,
        default_model="qwen-plus",
        openai_api_key="openai-key",
        dashscope_api_key="dashscope-key",
        anthropic_api_key="anthropic-key",
    )

    assert settings.llm_provider == PROVIDER_DASHSCOPE
    assert settings.llm_api_key == "dashscope-key"


def test_llm_api_key_does_not_fall_back_across_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    settings = Settings(
        _env_file=None,
        default_model="qwen-plus",
        openai_api_key="openai-key",
    )

    assert settings.llm_provider == PROVIDER_DASHSCOPE
    assert settings.llm_api_key is None
