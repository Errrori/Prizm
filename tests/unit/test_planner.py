import json

import pandas as pd
import pytest

from prizm.agent.planner import (
    SYSTEM_PROMPT,
    Planner,
    PlanParseError,
    build_mock_plan_response,
    build_plan_prompt,
    build_schema_summary,
    parse_plan,
)
from prizm.llm.adapter import MockLLM


def test_system_prompt_requires_month_format_and_scalar_shapes() -> None:
    assert "YYYY-MM" in SYSTEM_PROMPT
    assert "单行单列" in SYSTEM_PROMPT
    assert "多/少多少" in SYSTEM_PROMPT
    assert "在此基础上" in SYSTEM_PROMPT
    assert "COUNT(*)" in SYSTEM_PROMPT
    assert "最终 SELECT" in SYSTEM_PROMPT


def test_schema_summary_lists_columns_and_types() -> None:
    frame = pd.DataFrame({"channel": ["a"], "sales": [1]})

    summary = build_schema_summary(frame)

    assert "channel" in summary
    assert "sales" in summary


def test_prompt_injects_only_sample_rows() -> None:
    frame = pd.DataFrame({"id": range(50), "value": [f"v{i}" for i in range(50)]})

    messages = build_plan_prompt("有多少行？", "orders", frame, sample_rows=5)
    user_content = messages[1]["content"]

    assert "orders" in user_content
    assert "有多少行？" in user_content
    assert "v0" in user_content
    assert "v4" in user_content
    assert "v5" not in user_content
    assert "v49" not in user_content
    assert messages[0]["role"] == "system"


def test_prompt_uses_profile_summary_when_provided() -> None:
    frame = pd.DataFrame({"id": range(50), "value": [f"v{i}" for i in range(50)]})
    summary = "数据体检：50 行 × 2 列\n异常：未发现"

    messages = build_plan_prompt(
        "有多少行？",
        "orders",
        frame,
        profile_summary=summary,
    )
    user_content = messages[1]["content"]

    assert summary in user_content
    assert "列与类型" not in user_content
    assert "v49" not in user_content


def test_planner_passes_profile_summary_to_llm() -> None:
    frame = pd.DataFrame({"sales": [1, 2]})
    llm = MockLLM(json.dumps({"sql": "SELECT 1", "chart": None}))
    planner = Planner(llm)

    planner.plan("销售额", "orders", frame, profile_summary="数据体检：2 行 × 1 列")

    assert "数据体检：2 行 × 1 列" in llm.calls[0][1]["content"]


def test_prompt_injects_repair_feedback() -> None:
    frame = pd.DataFrame({"sales": [1, 2]})
    feedback = "第 1 次尝试失败。\n错误：BinderException: missing"

    messages = build_plan_prompt(
        "销售额",
        "orders",
        frame,
        profile_summary="数据体检：2 行 × 1 列",
        repair_feedback=feedback,
    )
    user_content = messages[1]["content"]

    assert "上一次尝试失败，需要修复" in user_content
    assert feedback in user_content
    assert "数据体检：2 行 × 1 列" in user_content


def test_planner_passes_repair_feedback_to_llm() -> None:
    frame = pd.DataFrame({"sales": [1, 2]})
    llm = MockLLM(json.dumps({"sql": "SELECT 1", "chart": None}))
    planner = Planner(llm)

    planner.plan("销售额", "orders", frame, repair_feedback="上次 SQL 缺列")

    assert "上次 SQL 缺列" in llm.calls[0][1]["content"]


def test_prompt_injects_conversation_summary() -> None:
    frame = pd.DataFrame({"sales": [1, 2]})
    summary = "第 1 轮问题：销售额\n生成的 SQL：SELECT SUM(sales) FROM orders"

    messages = build_plan_prompt(
        "只看已完成的再算一次",
        "orders",
        frame,
        profile_summary="数据体检：2 行 × 1 列",
        conversation_summary=summary,
    )
    user_content = messages[1]["content"]

    assert "最近对话与结果" in user_content
    assert summary in user_content
    assert "只看已完成的再算一次" in user_content


def test_planner_passes_conversation_summary_to_llm() -> None:
    frame = pd.DataFrame({"sales": [1, 2]})
    llm = MockLLM(json.dumps({"sql": "SELECT 1", "chart": None}))
    planner = Planner(llm)

    planner.plan("再算一次", "orders", frame, conversation_summary="上一轮 SQL：SELECT 1")

    assert "上一轮 SQL：SELECT 1" in llm.calls[0][1]["content"]


def test_parse_plan_accepts_fenced_json() -> None:
    text = '```json\n{"sql": "SELECT 1", "chart": null}\n```'

    plan = parse_plan(text)

    assert plan.sql == "SELECT 1"
    assert plan.chart is None


def test_parse_plan_accepts_chart_object() -> None:
    plan = parse_plan('{"sql": "SELECT 1", "chart": {"type": "bar"}}')

    assert plan.chart == {"type": "bar"}


@pytest.mark.parametrize(
    "text",
    [
        "没有 JSON",
        "{not json}",
        '{"chart": null}',
        '{"sql": "  "}',
        '{"sql": "SELECT 1", "chart": []}',
    ],
)
def test_parse_plan_rejects_invalid_output(text: str) -> None:
    with pytest.raises(PlanParseError):
        parse_plan(text)


def test_planner_uses_llm_response() -> None:
    frame = pd.DataFrame({"sales": [1, 2]})
    llm = MockLLM(json.dumps({"sql": "SELECT SUM(sales) FROM orders", "chart": None}))
    planner = Planner(llm)

    plan = planner.plan("销售额", "orders", frame)

    assert plan.sql == "SELECT SUM(sales) FROM orders"


def test_planner_rejects_empty_question() -> None:
    planner = Planner(MockLLM())

    with pytest.raises(ValueError, match="问题不能为空"):
        planner.plan("   ", "orders", pd.DataFrame({"a": [1]}))


def test_mock_plan_response_is_parseable_and_quoted() -> None:
    plan = parse_plan(build_mock_plan_response("orders"))

    assert plan.sql == 'SELECT * FROM "orders" LIMIT 5'
