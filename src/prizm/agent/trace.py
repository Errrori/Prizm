"""单轮分析的透明 trace：记录关键步骤、生成 SQL、执行状态与体检引用。

trace 只保存摘要与引用，不保存全量数据、样本行或密钥；供 Chainlit
步骤时间线和末尾透明面板展示“结论是怎么算出来的”。
"""

from __future__ import annotations

from dataclasses import dataclass

_ANOMALY_LABELS = {
    "exact_duplicate_rows": "完全重复行",
    "duplicate_primary_key": "主键重复",
    "missing_values": "缺失值",
    "dirty_numeric_format": "数值脏格式",
}

_POLICY_LABELS = {**_ANOMALY_LABELS, "ambiguous_question": "问题关注点"}

_EXECUTION_LABELS = {
    "succeeded": "执行成功",
    "repaired": "自修复后成功",
    "failed": "执行失败",
    "not_run": "未执行（等待澄清或走数据字典）",
}


def anomaly_label(code: str) -> str:
    return _ANOMALY_LABELS.get(code, code)


def policy_label(code: str) -> str:
    return _POLICY_LABELS.get(code, code)


def execution_label(status: str) -> str:
    return _EXECUTION_LABELS.get(status, status)


@dataclass(frozen=True, slots=True)
class TraceStep:
    """分析过程中的一个步骤。detail 用于步骤展开，summary 用于一行摘要。"""

    name: str
    status: str
    summary: str
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("trace 步骤名称不能为空")
        if not self.summary.strip():
            raise ValueError("trace 步骤摘要不能为空")


@dataclass(frozen=True, slots=True)
class ProfileReference:
    """本次分析实际引用的体检结论（只列触发澄清的异常）。"""

    code: str
    column: str | None
    count: int
    reason: str

    @property
    def label(self) -> str:
        return anomaly_label(self.code)


@dataclass(frozen=True, slots=True)
class Trace:
    question: str
    steps: tuple[TraceStep, ...] = ()
    generated_sql: str | None = None
    execution_status: str = "not_run"
    profile_references: tuple[ProfileReference, ...] = ()
    confirmed_policies: tuple[tuple[str, str], ...] = ()
    attempts: int = 0
    elapsed_ms: float | None = None

    def to_panel_markdown(self) -> str:
        """渲染末尾透明面板：状态 + SQL + 体检引用 + 步骤时间线。"""
        lines = ["### 透明面板", "", f"**执行状态**：{execution_label(self.execution_status)}"]
        if self.attempts:
            lines.append(f"**尝试次数**：{self.attempts}")
        if self.elapsed_ms is not None:
            lines.append(f"**耗时**：{self.elapsed_ms:.0f} ms")

        if self.generated_sql:
            lines.extend(["", "**生成的 SQL**", "```sql", self.generated_sql, "```"])

        lines.extend(["", "**引用了哪些体检结论**"])
        if self.profile_references:
            for reference in self.profile_references:
                column = f"（{reference.column}）" if reference.column else ""
                count = f"，{reference.count} 处" if reference.count else ""
                lines.append(
                    f"- {reference.label}{column}{count}：{reference.reason}"
                )
        else:
            lines.append("- 本次未触发异常口径澄清")

        if self.confirmed_policies:
            lines.extend(["", "**用户已确认口径**"])
            for name, choice in self.confirmed_policies:
                lines.append(f"- {name}：{choice}")

        if self.steps:
            lines.extend(["", "**分析步骤**"])
            for index, step in enumerate(self.steps, start=1):
                lines.append(f"{index}. [{step.status}] {step.name}：{step.summary}")

        return "\n".join(lines)
