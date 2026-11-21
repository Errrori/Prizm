"""LLM 用量统计与每日 token 额度台账。"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date


class QuotaExceededError(RuntimeError):
    """当日 token 额度已用完。"""


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} 必须是非负整数")
        if self.cost_usd is not None and self.cost_usd < 0:
            raise ValueError("cost_usd 不能为负数")
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            object.__setattr__(self, "total_tokens", self.prompt_tokens + self.completion_tokens)

    @classmethod
    def combine(cls, left: Usage, right: Usage) -> Usage:
        cost: float | None
        if left.cost_usd is None and right.cost_usd is None:
            cost = None
        else:
            cost = (left.cost_usd or 0.0) + (right.cost_usd or 0.0)
        return cls(
            prompt_tokens=left.prompt_tokens + right.prompt_tokens,
            completion_tokens=left.completion_tokens + right.completion_tokens,
            total_tokens=left.total_tokens + right.total_tokens,
            cost_usd=cost,
        )


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    usage: Usage = field(default_factory=Usage)


class SessionTurnCounter:
    """单会话提问次数计数；limit 为 0 表示不限制。"""

    def __init__(self, limit: int) -> None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit 必须是非负整数")
        self._limit = limit
        self._used = 0

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def used(self) -> int:
        return self._used

    def try_consume(self) -> bool:
        if self._limit > 0 and self._used >= self._limit:
            return False
        self._used += 1
        return True


class DailyUsageLedger:
    """按自然日累计 usage；跨日自动清零，线程安全。"""

    # DEBT-005: 额度台账仅存进程内存, 进程重启后清零。

    def __init__(self, *, today: Callable[[], date] | None = None) -> None:
        self._today = today or date.today
        self._lock = threading.Lock()
        self._day = self._today()
        self._usage = Usage()

    @property
    def day(self) -> date:
        with self._lock:
            self._roll(self._today())
            return self._day

    def snapshot(self) -> Usage:
        with self._lock:
            self._roll(self._today())
            return self._usage

    def record(self, usage: Usage) -> Usage:
        with self._lock:
            self._roll(self._today())
            self._usage = Usage.combine(self._usage, usage)
            return self._usage

    def remaining(self, budget: int) -> int:
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
            raise ValueError("budget 必须是非负整数")
        return max(0, budget - self.snapshot().total_tokens)

    def ensure_available(self, budget: int) -> None:
        if self.remaining(budget) <= 0:
            raise QuotaExceededError(f"今日 token 额度已用完（上限 {budget}）")

    def _roll(self, current: date) -> None:
        if current != self._day:
            self._day = current
            self._usage = Usage()
