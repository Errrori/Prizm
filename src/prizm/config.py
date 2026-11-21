"""全局配置：从环境变量 / .env 读取，密钥只存内存，不落盘、不打日志。"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MOCK_MODEL = "mock"


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

    @property
    def is_mock(self) -> bool:
        return self.default_model == MOCK_MODEL


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
