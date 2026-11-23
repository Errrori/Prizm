"""会话记忆：保存最近几轮问题、SQL 与结果摘要，供多轮追问理解指代。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MAX_TURNS = 5
DEFAULT_PROMPT_MAX_CHARS = 2_000
_MAX_PREVIEW_CHARS = 800


@dataclass(frozen=True, slots=True)
class Turn:
    question: str
    sql: str
    columns: tuple[str, ...] = ()
    row_count: int = 0
    preview: tuple[dict[str, Any], ...] = ()
    chart: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question 不能为空")
        if not self.sql.strip():
            raise ValueError("sql 不能为空")
        if (
            not isinstance(self.row_count, int)
            or isinstance(self.row_count, bool)
            or self.row_count < 0
        ):
            raise ValueError("row_count 必须是非负整数")


# DEBT-015: 只缓存结果摘要, 未把上一轮结果注册为可查询视图。
@dataclass
class ConversationMemory:
    """只保留最近 max_turns 轮，避免上下文无限增长。"""

    max_turns: int = DEFAULT_MAX_TURNS
    prompt_max_chars: int = DEFAULT_PROMPT_MAX_CHARS
    _turns: list[Turn] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_turns, int)
            or isinstance(self.max_turns, bool)
            or self.max_turns < 1
        ):
            raise ValueError("max_turns 必须是 >= 1 的整数")
        if (
            not isinstance(self.prompt_max_chars, int)
            or isinstance(self.prompt_max_chars, bool)
            or self.prompt_max_chars < 1
        ):
            raise ValueError("prompt_max_chars 必须是 >= 1 的整数")

    @property
    def turns(self) -> tuple[Turn, ...]:
        return tuple(self._turns)

    def __len__(self) -> int:
        return len(self._turns)

    def add(self, turn: Turn) -> None:
        self._turns.append(turn)
        if len(self._turns) > self.max_turns:
            del self._turns[: len(self._turns) - self.max_turns]

    def clear(self) -> None:
        self._turns.clear()

    def format_for_prompt(self, *, max_chars: int | None = None) -> str | None:
        """把最近轮次格式化为紧凑 prompt 片段；超长时优先保留最新轮次。"""
        limit = self.prompt_max_chars if max_chars is None else max_chars
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("max_chars 必须是 >= 1 的整数")
        if not self._turns:
            return None

        blocks = [
            self._format_turn(index, turn)
            for index, turn in enumerate(self._turns, 1)
        ]
        selected: list[str] = []
        remaining = limit
        for block in reversed(blocks):
            if len(block) <= remaining:
                selected.append(block)
                remaining -= len(block) + 2
                continue
            if not selected:
                selected.append(block[: max(0, remaining - 1)].rstrip() + "…")
            break
        return "\n\n".join(reversed(selected))

    @staticmethod
    def _format_turn(index: int, turn: Turn) -> str:
        columns = ", ".join(turn.columns) if turn.columns else "（无列）"
        preview = json.dumps(turn.preview, ensure_ascii=False, default=str)
        if len(preview) > _MAX_PREVIEW_CHARS:
            preview = preview[:_MAX_PREVIEW_CHARS] + "…"
        return (
            f"第 {index} 轮问题：{turn.question}\n"
            f"生成的 SQL：{turn.sql}\n"
            f"结果：{turn.row_count} 行；列：{columns}\n"
            f"结果前 {len(turn.preview)} 行：{preview}"
        )
