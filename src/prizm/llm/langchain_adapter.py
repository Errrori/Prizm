"""LangChain 适配层：用 ChatLiteLLM 实现自有 LLMAdapter 协议。

业务代码仍只依赖 ``prizm.llm.adapter`` 里的接口；本文件是另一个实现，
让 Planner / Composer / Repairer 无需改动即可接入 LangChain 生态。
``langchain_litellm`` 只在构建真实模型时惰性导入，避免测试阶段联网拉成本表。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from prizm.config import PROVIDER_KEY_ENV, Settings
from prizm.llm.adapter import (
    LLMAdapter,
    LLMConfigError,
    LLMError,
    Message,
    MockLLM,
    _configure_litellm_env,
)
from prizm.utils.tokens import Completion, Usage, UsageLedger

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

_ROLE_TO_KIND = {"system": "system", "user": "human", "assistant": "ai"}


def _to_lc_messages(messages: Sequence[Message]) -> list[BaseMessage]:
    """把自有 dict 消息转换成 LangChain 消息对象。"""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    classes = {"system": SystemMessage, "human": HumanMessage, "ai": AIMessage}
    converted: list[BaseMessage] = []
    for message in messages:
        kind = _ROLE_TO_KIND.get(str(message.get("role", "user")), "human")
        converted.append(classes[kind](content=str(message.get("content", ""))))
    return converted


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and value >= 0:
        return int(value)
    return 0


def extract_usage(message: Any) -> Usage:
    """从 LangChain 消息的 ``usage_metadata`` 提取 token；缺失按 0 处理。"""
    metadata = getattr(message, "usage_metadata", None)
    if not isinstance(metadata, Mapping):
        return Usage()
    return Usage(
        prompt_tokens=_as_int(metadata.get("input_tokens")),
        completion_tokens=_as_int(metadata.get("output_tokens")),
        total_tokens=_as_int(metadata.get("total_tokens")),
    )


def extract_text(message: Any) -> str:
    """兼容字符串内容和 LangChain 的多段 content 结构。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
                parts.append(str(block["text"]))
        return "".join(parts)
    return "" if content is None else str(content)


@dataclass
class LangChainLLMAdapter:
    """把 LangChain ChatModel 适配成项目内的 ``LLMAdapter`` 协议。"""

    chat: Any
    ledger: UsageLedger | None = None
    daily_token_budget: int | None = None

    def __post_init__(self) -> None:
        if self.daily_token_budget is not None and self.daily_token_budget < 0:
            raise ValueError("daily_token_budget 不能为负数")

    def _ensure_quota(self) -> None:
        if self.ledger is not None and self.daily_token_budget is not None:
            self.ledger.ensure_available(self.daily_token_budget)

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> Completion:
        self._ensure_quota()
        try:
            message = self.chat.invoke(_to_lc_messages(messages))
        except Exception as exc:
            raise LLMError(f"LLM 调用失败: {exc}") from exc

        usage = extract_usage(message)
        if self.ledger is not None:
            self.ledger.record(usage)

        text = extract_text(message)
        if not text.strip():
            raise LLMError("LLM 返回内容为空")
        return Completion(text, usage)

    def stream(
        self, messages: Sequence[Message], *, temperature: float = 0.0
    ) -> Iterator[str]:
        """逐块产出文本；usage 取流中最后一条带 usage 的消息。"""
        self._ensure_quota()
        usage: Usage | None = None
        emitted_chars = 0
        try:
            for chunk in self.chat.stream(_to_lc_messages(messages)):
                text = extract_text(chunk)
                if text:
                    emitted_chars += len(text)
                    yield text
                chunk_usage = extract_usage(chunk)
                if (
                    chunk_usage.prompt_tokens
                    or chunk_usage.completion_tokens
                    or chunk_usage.total_tokens
                ):
                    usage = chunk_usage
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"LLM 流式调用失败: {exc}") from exc
        finally:
            if self.ledger is not None and usage is not None:
                self.ledger.record(usage)
        if emitted_chars == 0:
            raise LLMError("LLM 返回内容为空")


def build_langchain_llm(
    settings: Settings,
    *,
    chat: Any | None = None,
    response: str | None = None,
    ledger: UsageLedger | None = None,
    daily_token_budget: int | None = None,
) -> LLMAdapter:
    """按配置构建 LangChain 适配器；mock 模式不触网，测试可注入 chat。"""
    if settings.is_mock:
        return MockLLM() if response is None else MockLLM(response)
    if chat is not None:
        return LangChainLLMAdapter(chat, ledger=ledger, daily_token_budget=daily_token_budget)

    provider = settings.llm_provider
    api_key = settings.llm_api_key
    if not api_key:
        env_name = PROVIDER_KEY_ENV[provider]
        raise LLMConfigError(
            f"模型 {settings.default_model} 属于 {provider}，需要环境变量 {env_name}，当前未配置"
        )

    _configure_litellm_env()
    from langchain_litellm import ChatLiteLLM

    model = ChatLiteLLM(
        model=settings.default_model,
        api_key=api_key,
        api_base=settings.llm_api_base,
        temperature=0.0,
        max_retries=settings.llm_max_retries,
    )
    budget = daily_token_budget if daily_token_budget is not None else settings.daily_token_budget
    return LangChainLLMAdapter(
        model,
        ledger=ledger,
        daily_token_budget=budget if ledger is not None else None,
    )
