import pytest

from prizm.agent.trace import (
    ProfileReference,
    Trace,
    TraceStep,
    anomaly_label,
    execution_label,
    policy_label,
)


def test_panel_renders_sql_status_references_and_policies() -> None:
    trace = Trace(
        question="统计销售额",
        steps=(
            TraceStep("体检与澄清判定", "ok", "无新增待澄清项"),
            TraceStep("执行", "ok", "返回 1 行"),
        ),
        generated_sql="SELECT 1 AS ok",
        execution_status="repaired",
        profile_references=(
            ProfileReference(
                code="dirty_numeric_format",
                column="sales",
                count=5,
                reason="列 sales 有 5 个值混入千分位/单位",
            ),
        ),
        confirmed_policies=(("数值脏格式", "清洗为数值后统计"),),
        attempts=2,
        elapsed_ms=12.6,
    )

    panel = trace.to_panel_markdown()

    assert "**执行状态**：自修复后成功" in panel
    assert "**尝试次数**：2" in panel
    assert "**耗时**：13 ms" in panel
    assert "SELECT 1 AS ok" in panel
    assert "数值脏格式（sales），5 处" in panel
    assert "数值脏格式：清洗为数值后统计" in panel
    assert "1. [ok] 体检与澄清判定：无新增待澄清项" in panel


def test_panel_without_references_and_sql_marks_not_triggered() -> None:
    trace = Trace(question="随便看看", execution_status="succeeded")

    panel = trace.to_panel_markdown()

    assert "**执行状态**：执行成功" in panel
    assert "本次未触发异常口径澄清" in panel
    assert "生成的 SQL" not in panel
    assert "用户已确认口径" not in panel


def test_trace_step_requires_name_and_summary() -> None:
    with pytest.raises(ValueError, match="名称"):
        TraceStep("", "ok", "摘要")
    with pytest.raises(ValueError, match="摘要"):
        TraceStep("执行", "ok", "")


def test_labels_fall_back_to_code() -> None:
    assert anomaly_label("missing_values") == "缺失值"
    assert policy_label("ambiguous_question") == "问题关注点"
    assert policy_label("unknown_code") == "unknown_code"
    assert execution_label("unknown_status") == "unknown_status"
