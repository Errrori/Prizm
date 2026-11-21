"""全局配置：从环境变量 / .env 读取，密钥只存内存，不落盘、不打日志。"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MOCK_MODEL = "mock"

PROVIDER_OPENAI = "openai"
PROVIDER_DASHSCOPE = "dashscope"
PROVIDER_ANTHROPIC = "anthropic"

PROVIDER_KEY_ENV = {
    PROVIDER_OPENAI: "OPENAI_API_KEY",
    PROVIDER_DASHSCOPE: "DASHSCOPE_API_KEY",
    PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY",
}


def detect_provider(model: str) -> str:
    """按模型名判定 provider；未识别的按 OpenAI 兼容处理。"""
    name = model.strip().lower()
    if name.startswith("dashscope/") or name.startswith("qwen") or "dashscope" in name:
        return PROVIDER_DASHSCOPE
    if name.startswith("anthropic/") or name.startswith("claude") or "anthropic" in name:
        return PROVIDER_ANTHROPIC
    return PROVIDER_OPENAI


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    default_model: str = Field(default=MOCK_MODEL, validation_alias="PRIZM_DEFAULT_MODEL")
    daily_token_budget: int = Field(default=200_000, validation_alias="PRIZM_DAILY_TOKEN_BUDGET")
    session_turn_limit: int = Field(default=30, validation_alias="PRIZM_SESSION_TURN_LIMIT")

    duckdb_memory_limit: str = Field(default="1GB", validation_alias="PRIZM_DUCKDB_MEMORY_LIMIT")
    duckdb_threads: int = Field(default=2, validation_alias="PRIZM_DUCKDB_THREADS")
    sql_timeout_seconds: float = Field(default=30.0, validation_alias="PRIZM_SQL_TIMEOUT_SECONDS")

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    dashscope_api_key: str | None = Field(default=None, validation_alias="DASHSCOPE_API_KEY")
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    llm_api_base: str | None = Field(default=None, validation_alias="PRIZM_LLM_API_BASE")

    @property
    def is_mock(self) -> bool:
        return self.default_model == MOCK_MODEL

    @property
    def llm_provider(self) -> str:
        return detect_provider(self.default_model)

    @property
    def llm_api_key(self) -> str | None:
        """只取当前 provider 对应的 key，不做跨 provider 回退。"""
        keys = {
            PROVIDER_OPENAI: self.openai_api_key,
            PROVIDER_DASHSCOPE: self.dashscope_api_key,
            PROVIDER_ANTHROPIC: self.anthropic_api_key,
        }
        return keys[self.llm_provider]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
