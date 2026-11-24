"""把自然语言问题规划为只读 SQL 与可选 chart spec。

不执行、不生成自由 Python；图表只输出结构化 spec，由渲染层安全绘制。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from prizm.engine.duckdb_engine import quote_identifier
from prizm.llm.adapter import LLMAdapter, Message

# DEBT-026: 差值类问题依赖 prompt 约束, 未做确定性 SQL 结构校验。
SYSTEM_PROMPT = """你是数据分析 SQL 规划器，只输出一个 JSON 对象，不要输出解释或 Markdown。
JSON 结构：{"sql": "<DuckDB 只读 SQL>", "chart": null}
或：{"sql": "<DuckDB 只读 SQL>", "chart": {"type": "bar|line|pie|scatter",
"x": "<列名>", "y": "<列名>", "title": "<可选标题>"}}
规则：
- 只能生成单条 SELECT 或 WITH ... SELECT 语句，禁止任何写操作或 DDL。
- 只能使用给定 schema 中存在的列，不要臆造列名。
- 标识符必要时用双引号包裹。
- 体检摘要中的异常仅作上下文提示，不要据此臆造数据或擅自改变统计口径。
- 若提供了最近对话与结果，结合上一轮问题和 SQL 理解“上一题/再算一次”等指代。
- 若提供了用户已确认的数据口径，必须严格遵循，不得自行更改。
- 若提供了上次失败信息，必须针对错误修正 SQL，不要重复同样的错误。
- 聚合结果请把聚合列别名设为原列名：SUM(sales) AS sales、AVG(sales) AS sales、
  SUM(revenue) AS revenue；计数列命名为 count；时间分组用 month。
- 时间列可能是 VARCHAR：按月份分组时不要直接 EXTRACT(... FROM 月份列)；
  优先用 substr(月份列, 1, 7) AS month 得到 YYYY-MM，或先 CAST(月份列 AS DATE/TIMESTAMP) 再分组。
  按月份统计必须输出 YYYY-MM 格式的 month 列。
- 问“订单/记录数量”用 COUNT(*)，不要用 SUM(quantity)；
  只有明确问“销量/件数/units_sold”时才求和数量列。
- 比较 DATE/TIMESTAMP 列时不要直接与 'YYYY-MM' 比较；按月统计统一用 substr(..., 1, 7)，
  需要日期边界时用 'YYYY-MM-01' 或 DATE 'YYYY-MM-DD'。
- 问“A 比 B 多/少多少”时，只返回单行单列的差值，方向必须是前者减后者（A - B）；
  例如“12 月比 1 月多多少”= 12 月 - 1 月，不要反过来。
  （如 SUM(CASE WHEN channel='A' THEN sales END)
   - SUM(CASE WHEN channel='B' THEN sales END) AS diff），
  禁止返回按 A/B 分组的明细。
- 问“总量/合计/总额/总计”时，只返回单行单列的聚合值，禁止保留 GROUP BY 明细。
- 问“数量/多少”且已限定对象（如“该地区”“刚才那个渠道”）时，只返回该聚合值一列，
  不要附加对象名称列；问“哪个/哪个月”且只要求名称或月份时，只返回该列。
- 问“成本/销量等指标最高的前 N 个 X 的某指标总量”时，先用子查询或 CTE 按该指标选出前 N 个 X，
  再对其求和，最终只返回单行单列。
- 使用 WITH/子查询时必须写出最终 SELECT，禁止只输出 CTE 定义；
  所有列引用和聚合函数都必须有对应的 FROM 表。
- 多轮追问“在此基础上/再看/只保留”时，必须沿用上一轮的筛选条件和聚合口径，
  再按本轮要求追加过滤或聚合。
- 示例（按月趋势）：SELECT substr(<时间列>, 1, 7) AS month, SUM(<数值列>) AS <数值列>
  FROM <表> WHERE <过滤条件> GROUP BY month ORDER BY month
- 示例（差值）：SELECT SUM(CASE WHEN substr(<时间列>, 1, 7) = '<月份A>' THEN <数值列> ELSE 0 END)
  - SUM(CASE WHEN substr(<时间列>, 1, 7) = '<月份B>' THEN <数值列> ELSE 0 END) AS diff
  FROM <表> WHERE <过滤条件>
- 问“是否存在完全重复的行 / 重复行数”时，必须返回一行计数
  （如 COUNT(*) - COUNT(DISTINCT <所有列>) AS duplicate_rows），
  不能返回 GROUP BY ... HAVING 的空结果。
- chart 可选，只有结果适合图表时才提供：x 为分类/时间列，y 为数值列，且都必须是 SQL 结果中的列名。
- bar 用于分类对比，line 用于趋势，pie 用于占比（y 必须非负），scatter 用于两个数值列的关系。
- 单值汇总、多列明细或无法确定合适坐标轴时，chart 必须为 null。"""


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
