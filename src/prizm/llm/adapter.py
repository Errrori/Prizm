"""LLM 适配层：业务代码只依赖本接口，litellm 仅在本文件内出现。

阶段 1 默认使用 :class:`MockLLM`，无需 key 即可跑通链路；
配置真实模型后由 :func:`build_llm` 切换到 :class:`LiteLLMAdapter`。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from prizm.config import PROVIDER_KEY_ENV, Settings
from prizm.utils.tokens import Completion, Usage, UsageLedger

Message = dict[str, str]
CompletionFn = Callable[..., Any]


class LLMError(RuntimeError):
    """LLM 调用失败或返回结构非法。"""


class LLMConfigError(LLMError):
    """模型配置不完整或不匹配，启动阶段即失败。"""


class LLMAdapter(Protocol):
    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> Completion: ...


@dataclass
class MockLLM:
    """固定返回一段文本，用于无 key 的开发与单测。"""

    response: str = '{"sql": "SELECT 1 AS ok", "chart": null}'
    calls: list[list[Message]] = field(default_factory=list)

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> Completion:
        self.calls.append([dict(message) for message in messages])
        return Completion(self.response)


def _read_field(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _as_token_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and value >= 0:
        return int(value)
    return 0


def extract_usage(response: Any) -> Usage:
    """从 litellm 响应中提取 token 与费用；缺失字段按 0 处理。"""
    raw_usage = _read_field(response, "usage")
    # DEBT-006: 费用依赖 litellm 响应的 _hidden_params.response_cost, 缺失时无法统计。
    cost = _read_field(_read_field(response, "_hidden_params"), "response_cost")
    cost_usd = (
        float(cost)
        if isinstance(cost, (int, float)) and not isinstance(cost, bool)
        else None
    )
    return Usage(
        prompt_tokens=_as_token_count(_read_field(raw_usage, "prompt_tokens")),
        completion_tokens=_as_token_count(_read_field(raw_usage, "completion_tokens")),
        total_tokens=_as_token_count(_read_field(raw_usage, "total_tokens")),
        cost_usd=cost_usd,
    )


def _configure_litellm_env() -> None:
    """隔离 litellm 的进程默认值：不读 CWD .env，也不联网拉成本表。"""
    os.environ.setdefault("LITELLM_MODE", "PROD")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


@dataclass
class LiteLLMAdapter:
    """litellm 封装；completion_fn 可在测试中注入，避免网络与真实 key。"""

    model: str
    api_key: str | None = None
    api_base: str | None = None
    completion_fn: CompletionFn | None = None
    ledger: UsageLedger | None = None
    daily_token_budget: int | None = None
    max_retries: int = 0

    def __post_init__(self) -> None:
        if self.daily_token_budget is not None and self.daily_token_budget < 0:
            raise ValueError("daily_token_budget 不能为负数")
        if self.max_retries < 0:
            raise ValueError("max_retries 不能为负数")

    def _completion(self) -> CompletionFn:
        if self.completion_fn is not None:
            return self.completion_fn
        _configure_litellm_env()
        from litellm import completion

        return cast(CompletionFn, completion)

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> Completion:
        if self.ledger is not None and self.daily_token_budget is not None:
            self.ledger.ensure_available(self.daily_token_budget)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.max_retries > 0:
            kwargs["num_retries"] = self.max_retries
        try:
            response = self._completion()(**kwargs)
        except Exception as exc:
            raise LLMError(f"LLM 调用失败: {exc}") from exc

        usage = extract_usage(response)
        if self.ledger is not None:
            self.ledger.record(usage)

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise LLMError("LLM 返回结构非法") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM 返回内容为空")
        return Completion(content, usage)


def build_llm(
    settings: Settings,
    *,
    response: str | None = None,
    ledger: UsageLedger | None = None,
    daily_token_budget: int | None = None,
) -> LLMAdapter:
    """按配置构建适配器；mock 模式不触网。

    真实模型要求当前 provider 的 key 已配置，且 api_base 显式由配置传入，
    避免依赖 litellm 隐式读取 .env。传入 ledger 时按每日额度记账。
    """
    if settings.is_mock:
        return MockLLM() if response is None else MockLLM(response)

    provider = settings.llm_provider
    api_key = settings.llm_api_key
    if not api_key:
        env_name = PROVIDER_KEY_ENV[provider]
        raise LLMConfigError(
            f"模型 {settings.default_model} 属于 {provider}，需要环境变量 {env_name}，当前未配置"
        )
    budget = daily_token_budget if daily_token_budget is not None else settings.daily_token_budget
    return LiteLLMAdapter(
        settings.default_model,
        api_key=api_key,
        api_base=settings.llm_api_base,
        ledger=ledger,
        daily_token_budget=budget if ledger is not None else None,
        max_retries=settings.llm_max_retries,
    )
