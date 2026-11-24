"""编排过程进度事件：供 UI 实时渲染步骤与结论 token，不参与分析决策。

编排本身保持同步，UI 通过回调把事件桥接到异步渲染层；
不传回调时事件不会被产生，行为与原来完全一致。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

ProgressKind = Literal["step", "conclusion_token"]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    kind: ProgressKind
    name: str = ""
    status: str = ""
    summary: str = ""
    detail: str | None = None
    text: str = ""


ProgressSink = Callable[[ProgressEvent], None]
TokenSink = Callable[[str], None]
