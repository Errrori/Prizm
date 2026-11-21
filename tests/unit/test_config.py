import pytest

from prizm.config import MOCK_MODEL, Settings


def test_settings_default_to_mock_and_limits() -> None:
    settings = Settings(_env_file=None)

    assert settings.default_model == MOCK_MODEL
    assert settings.is_mock is True
    assert settings.duckdb_memory_limit == "1GB"
    assert settings.duckdb_threads == 2
    assert settings.sql_timeout_seconds == 30.0
    assert settings.openai_api_key is None


def test_settings_read_prefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIZM_DEFAULT_MODEL", "qwen-plus")
    monkeypatch.setenv("PRIZM_DUCKDB_THREADS", "4")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert settings.default_model == "qwen-plus"
    assert settings.is_mock is False
    assert settings.duckdb_threads == 4
    assert settings.dashscope_api_key == "test-key"
