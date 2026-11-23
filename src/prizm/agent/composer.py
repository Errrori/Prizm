"""结论生成：优先让 LLM 根据结果归纳，失败时回退确定性模板。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from prizm.llm.adapter import LLMAdapter, LLMError, Message
from prizm.utils.tokens import QuotaExceededError

if TYPE_CHECKING:
    from prizm.viz.charts import ChartSpec

_SYSTEM_PROMPT = """你是数据分析结论撰写助手，只基于给定问题和查询结果写结论。
要求：
- 用中文写 1-3 句话，直接给结论，不要复述 SQL，不要输出 Markdown 或代码块。
- 只能引用结果中出现的数字与字段，禁止编造。
- 结果为空时如实说明没有数据。
- 不做超出数据范围的推测或建议。"""

_MAX_PREVIEW_ROWS = 10
_MAX_PROMPT_CHARS = 4000
_MAX_CONCLUSION_CHARS = 300
_FENCE_RE = re.compile(r"^```[A-Za-z]*\s*|\s*```$")


@dataclass(frozen=True, slots=True)
class Conclusion:
    text: str
    source: str = "template"
    fallback_reason: str | None = None


def _format_value(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text[:80] + "…" if len(text) > 80 else text


def template_conclusion(frame: pd.DataFrame) -> str:
    """不依赖模型的结果摘要，作为 LLM 失败时的兜底。"""
    if frame.empty:
        return "查询结果为空，未返回任何数据行。"
    if len(frame) == 1 and len(frame.columns) == 1:
        column = str(frame.columns[0])
        return f"{column} = {_format_value(frame.iloc[0, 0])}。"

    parts = [f"共返回 {len(frame)} 行 × {len(frame.columns)} 列"]
    numeric = frame.select_dtypes(include="number")
    if not numeric.empty:
        column = str(numeric.columns[0])
        parts.append(f"{column} 合计 {_format_value(numeric[column].sum())}")
    names = "、".join(str(column) for column in frame.columns[:6])
    if len(frame.columns) > 6:
        names += " 等"
    parts.append(f"列：{names}")
    return "；".join(parts) + "。"


def build_conclusion_prompt(
    question: str,
    sql: str,
    frame: pd.DataFrame,
    *,
    chart: ChartSpec | None = None,
) -> list[Message]:
    """只注入结果规模与前若干行样本，绝不注入全量数据。"""
    preview = frame.head(_MAX_PREVIEW_ROWS)
    records = [
        {str(key): _format_value(value) for key, value in row.items()}
        for row in preview.to_dict(orient="records")
    ]
    sample = json.dumps(records, ensure_ascii=False)
    if len(sample) > _MAX_PROMPT_CHARS:
        sample = sample[:_MAX_PROMPT_CHARS] + "…（样本已截断）"

    chart_line = ""
    if chart is not None:
        chart_line = f"\n图表：{chart.type}（x={chart.x}, y={chart.y}）"

    user_prompt = (
        f"问题：{question}\n"
        f"生成 SQL：{sql}\n"
        f"结果规模：{len(frame)} 行 × {len(frame.columns)} 列\n"
        f"前 {len(preview)} 行样本（JSON）：\n{sample}"
        f"{chart_line}"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def clean_conclusion(text: str) -> str:
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    if not cleaned:
        raise ValueError("LLM 未返回结论")
    if len(cleaned) > _MAX_CONCLUSION_CHARS:
        cleaned = cleaned[:_MAX_CONCLUSION_CHARS].rstrip() + "…"
    return cleaned


@dataclass
class ConclusionComposer:
    # DEBT-023: 结论质量未纳入评测, 模型失败或额度不足时静默回退模板。
    llm: LLMAdapter

    def compose(
        self,
        question: str,
        sql: str,
        frame: pd.DataFrame,
        *,
        chart: ChartSpec | None = None,
    ) -> Conclusion:
        try:
            completion = self.llm.complete(
                build_conclusion_prompt(question, sql, frame, chart=chart)
            )
            return Conclusion(clean_conclusion(completion.text), source="llm")
        except (LLMError, QuotaExceededError, ValueError) as exc:
            return Conclusion(
                template_conclusion(frame),
                source="template",
                fallback_reason=str(exc) or exc.__class__.__name__,
            )
