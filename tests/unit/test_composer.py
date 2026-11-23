from collections.abc import Sequence

import pandas as pd
import pytest

from prizm.agent.composer import (
    ConclusionComposer,
    build_conclusion_prompt,
    clean_conclusion,
    template_conclusion,
)
from prizm.llm.adapter import Completion, LLMError, Message, MockLLM
from prizm.utils.tokens import QuotaExceededError
from prizm.viz.charts import ChartSpec


class FailingLLM:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> Completion:
        self.calls.append([dict(message) for message in messages])
        raise self.error


def test_template_conclusion_handles_empty() -> None:
    assert template_conclusion(pd.DataFrame()) == "查询结果为空，未返回任何数据行。"


def test_template_conclusion_formats_single_value() -> None:
    frame = pd.DataFrame({"total": [300.0]})

    assert template_conclusion(frame) == "total = 300。"


def test_template_conclusion_summarizes_multiple_rows() -> None:
    frame = pd.DataFrame({"渠道": ["线上", "线下"], "销售额": [100, 200]})

    text = template_conclusion(frame)

    assert "2 行" in text
    assert "销售额 合计 300" in text
    assert "渠道、销售额" in text


def test_conclusion_prompt_injects_only_sample_rows() -> None:
    frame = pd.DataFrame({"id": range(50), "value": [f"v{i}" for i in range(50)]})

    messages = build_conclusion_prompt("有多少行？", "SELECT 1", frame)
    user_content = messages[1]["content"]

    assert "有多少行？" in user_content
    assert "SELECT 1" in user_content
    assert "结果规模：50 行 × 2 列" in user_content
    assert "v0" in user_content
    assert "v9" in user_content
    assert "v10" not in user_content
    assert "v49" not in user_content


def test_conclusion_prompt_includes_chart_metadata() -> None:
    frame = pd.DataFrame({"渠道": ["线上"], "销售额": [100]})
    chart = ChartSpec(type="bar", x="渠道", y="销售额")

    messages = build_conclusion_prompt("按渠道统计", "SELECT 1", frame, chart=chart)

    assert "图表：bar（x=渠道, y=销售额）" in messages[1]["content"]


def test_clean_conclusion_strips_code_fence_and_truncates() -> None:
    assert clean_conclusion("```\n销售额合计 300。\n```") == "销售额合计 300。"

    long_text = "结" * 400
    cleaned = clean_conclusion(long_text)

    assert len(cleaned) == 301
    assert cleaned.endswith("…")


def test_clean_conclusion_rejects_empty() -> None:
    with pytest.raises(ValueError, match="未返回结论"):
        clean_conclusion("   ")


def test_composer_uses_llm_text() -> None:
    llm = MockLLM("已完成订单的销售额合计为 300。")
    composer = ConclusionComposer(llm)
    frame = pd.DataFrame({"total": [300]})

    conclusion = composer.compose("已完成销售额", "SELECT 1", frame)

    assert conclusion.text == "已完成订单的销售额合计为 300。"
    assert conclusion.source == "llm"
    assert conclusion.fallback_reason is None
    assert "已完成销售额" in llm.calls[0][1]["content"]


@pytest.mark.parametrize(
    "error",
    [LLMError("模型超时"), QuotaExceededError("额度用完"), ValueError("解析失败")],
)
def test_composer_falls_back_on_errors(error: Exception) -> None:
    frame = pd.DataFrame({"total": [300]})
    composer = ConclusionComposer(FailingLLM(error))

    conclusion = composer.compose("已完成销售额", "SELECT 1", frame)

    assert conclusion.source == "template"
    assert conclusion.text == "total = 300。"
    assert conclusion.fallback_reason


def test_composer_falls_back_on_empty_llm_text() -> None:
    frame = pd.DataFrame({"total": [300]})
    composer = ConclusionComposer(MockLLM("   "))

    conclusion = composer.compose("已完成销售额", "SELECT 1", frame)

    assert conclusion.source == "template"
    assert conclusion.fallback_reason == "LLM 未返回结论"
