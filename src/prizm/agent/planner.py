"""把自然语言问题规划为只读 SQL 与可选 chart spec。

本阶段只产出 SQL，不执行、不生成自由 Python；chart 固定为 null。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from prizm.engine.duckdb_engine import quote_identifier
from prizm.llm.adapter import LLMAdapter, Message

SYSTEM_PROMPT = """你是数据分析 SQL 规划器，只输出一个 JSON 对象，不要输出解释或 Markdown。
JSON 结构：{"sql": "<DuckDB 只读 SQL>", "chart": null}
规则：
- 只能生成单条 SELECT 或 WITH ... SELECT 语句，禁止任何写操作或 DDL。
- 只能使用给定 schema 中存在的列，不要臆造列名。
- 标识符必要时用双引号包裹。
- 体检摘要中的异常仅作上下文提示，不要据此臆造数据或擅自改变统计口径。
- 若提供了最近对话与结果，结合上一轮问题和 SQL 理解“上一题/再算一次”等指代。
- 若提供了用户已确认的数据口径，必须严格遵循，不得自行更改。
- 若提供了上次失败信息，必须针对错误修正 SQL，不要重复同样的错误。
- 本阶段不绘图，chart 必须为 null。"""


class PlanParseError(ValueError):
    """LLM 输出无法解析成执行计划。"""


@dataclass(frozen=True, slots=True)
class Plan:
    sql: str
    chart: dict[str, Any] | None = None


def build_mock_plan_response(view_name: str) -> str:
    """开发/演示用：不调用真实模型，返回“查看前 5 行”的合法计划。"""
    sql = f"SELECT * FROM {quote_identifier(view_name)} LIMIT 5"
    return json.dumps({"sql": sql, "chart": None}, ensure_ascii=False)


def build_schema_summary(frame: pd.DataFrame) -> str:
    return "\n".join(f"- {column} ({dtype})" for column, dtype in frame.dtypes.items())


def build_plan_prompt(
    question: str,
    view_name: str,
    frame: pd.DataFrame,
    *,
    sample_rows: int = 5,
    profile_summary: str | None = None,
    confirmed_policies: Mapping[str, str] | None = None,
    conversation_summary: str | None = None,
    repair_feedback: str | None = None,
) -> list[Message]:
    """构造规划 prompt：只注入 schema + 体检摘要或前若干行样本，绝不注入全量数据。"""
    if profile_summary:
        context = f"数据体检摘要：\n{profile_summary}"
    else:
        sample = frame.head(sample_rows).to_dict(orient="records")
        context = (
            f"列与类型：\n{build_schema_summary(frame)}\n"
            f"前 {sample_rows} 行样本（JSON）：\n"
            f"{json.dumps(sample, ensure_ascii=False, default=str)}"
        )
    if confirmed_policies:
        policies = "\n".join(f"- {name}：{choice}" for name, choice in confirmed_policies.items())
        context += f"\n\n用户已确认的数据口径（必须遵守）：\n{policies}"
    if conversation_summary:
        context += f"\n\n最近对话与结果（用于理解指代，不要臆造）：\n{conversation_summary}"
    if repair_feedback:
        context += f"\n\n上一次尝试失败，需要修复：\n{repair_feedback}"
    user_prompt = f"数据集视图：{view_name}\n{context}\n问题：{question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_plan(text: str) -> Plan:
    raw = text.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end < start:
        raise PlanParseError("LLM 未返回 JSON 对象")

    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise PlanParseError(f"LLM 返回的 JSON 解析失败: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise PlanParseError("LLM 返回的 JSON 必须是对象")

    sql = payload.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        raise PlanParseError("计划缺少非空 sql 字段")

    chart = payload.get("chart")
    if chart is not None and not isinstance(chart, dict):
        raise PlanParseError("chart 必须是对象或 null")

    return Plan(sql=sql.strip(), chart=chart)


@dataclass
class Planner:
    llm: LLMAdapter
    sample_rows: int = 5

    def plan(
        self,
        question: str,
        view_name: str,
        frame: pd.DataFrame,
        *,
        profile_summary: str | None = None,
        confirmed_policies: Mapping[str, str] | None = None,
        conversation_summary: str | None = None,
        repair_feedback: str | None = None,
    ) -> Plan:
        if not question.strip():
            raise ValueError("问题不能为空")
        messages = build_plan_prompt(
            question,
            view_name,
            frame,
            sample_rows=self.sample_rows,
            profile_summary=profile_summary,
            confirmed_policies=confirmed_policies,
            conversation_summary=conversation_summary,
            repair_feedback=repair_feedback,
        )
        return parse_plan(self.llm.complete(messages).text)
