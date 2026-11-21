"""LLM 适配层：业务代码只依赖本接口，litellm 仅在本文件内出现。

阶段 1 默认使用 :class:`MockLLM`，无需 key 即可跑通链路；
配置真实模型后由 :func:`build_llm` 切换到 :class:`LiteLLMAdapter`。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from prizm.config import Settings

Message = dict[str, str]
CompletionFn = Callable[..., Any]


class LLMError(RuntimeError):
    """LLM 调用失败或返回结构非法。"""


class LLMAdapter(Protocol):
    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> str: ...


@dataclass
class MockLLM:
    """固定返回一段文本，用于无 key 的开发与单测。"""

    response: str = '{"sql": "SELECT 1 AS ok", "chart": null}'
    calls: list[list[Message]] = field(default_factory=list)

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> str:
        self.calls.append([dict(message) for message in messages])
        return self.response


@dataclass
class LiteLLMAdapter:
    """litellm 封装；completion_fn 可在测试中注入，避免网络与真实 key。"""

    model: str
    api_key: str | None = None
    completion_fn: CompletionFn | None = None

    def _completion(self) -> CompletionFn:
        if self.completion_fn is not None:
            return self.completion_fn
        from litellm import completion

        return cast(CompletionFn, completion)

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> str:
        # DEBT-002: 未统计 token/费用, 也未执行每日额度与会话次数上限。
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        try:
            response = self._completion()(**kwargs)
        except Exception as exc:
            raise LLMError(f"LLM 调用失败: {exc}") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise LLMError("LLM 返回结构非法") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM 返回内容为空")
        return content


def build_llm(settings: Settings, *, response: str | None = None) -> LLMAdapter:
    """按配置构建适配器；mock 模式不触网。"""
    if settings.is_mock:
        return MockLLM() if response is None else MockLLM(response)

    api_key = settings.dashscope_api_key or settings.openai_api_key or settings.anthropic_api_key
    return LiteLLMAdapter(settings.default_model, api_key=api_key)
