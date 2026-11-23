import pandas as pd
import pytest

from prizm.viz.charts import (
    ChartError,
    ChartSpec,
    parse_chart_spec,
    render_chart,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"渠道": ["线上", "线下"], "销售额": [100, 200]})


def test_parse_chart_spec_accepts_valid_spec() -> None:
    spec = parse_chart_spec(
        {"type": "Bar", "x": "渠道", "y": "销售额", "title": "  渠道对比  "},
        ["渠道", "销售额"],
    )

    assert spec == ChartSpec(type="bar", x="渠道", y="销售额", title="渠道对比")
    assert spec is not None
    assert spec.as_dict() == {
        "type": "bar",
        "x": "渠道",
        "y": "销售额",
        "title": "渠道对比",
    }


def test_parse_chart_spec_treats_null_and_blank_title(frame: pd.DataFrame) -> None:
    assert parse_chart_spec(None, frame.columns) is None
    spec = parse_chart_spec({"type": "line", "x": "渠道", "y": "销售额"}, frame.columns)

    assert spec is not None
    assert spec.title is None


@pytest.mark.parametrize("kind", ["", "area", "table", None, 1])
def test_parse_chart_spec_rejects_unknown_type(kind: object) -> None:
    with pytest.raises(ChartError, match=r"chart\.type"):
        parse_chart_spec({"type": kind, "x": "渠道", "y": "销售额"}, ["渠道", "销售额"])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"type": "bar", "y": "销售额"}, r"chart\.x"),
        ({"type": "bar", "x": "渠道"}, r"chart\.y"),
        ({"type": "bar", "x": "缺失", "y": "销售额"}, "不在结果列中"),
        ({"type": "bar", "x": "渠道", "y": "销售额", "title": 1}, r"chart\.title"),
        ([], "必须是对象"),
    ],
)
def test_parse_chart_spec_rejects_invalid_fields(payload: object, message: str) -> None:
    with pytest.raises(ChartError, match=message):
        parse_chart_spec(payload, ["渠道", "销售额"])  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["bar", "line", "pie", "scatter"])
def test_render_chart_returns_png(kind: str, frame: pd.DataFrame) -> None:
    chart_frame = frame
    if kind == "scatter":
        chart_frame = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
        spec = ChartSpec(type=kind, x="x", y="y")
    else:
        spec = ChartSpec(type=kind, x="渠道", y="销售额")

    png = render_chart(spec, chart_frame)

    assert png.startswith(PNG_SIGNATURE)
    assert len(png) > 1000


def test_render_chart_rejects_empty_result(frame: pd.DataFrame) -> None:
    spec = ChartSpec(type="bar", x="渠道", y="销售额")

    with pytest.raises(ChartError, match="数据为空"):
        render_chart(spec, frame.iloc[0:0])


def test_render_chart_rejects_non_numeric_y() -> None:
    frame = pd.DataFrame({"渠道": ["线上"], "销售额": ["不是数字"]})
    spec = ChartSpec(type="bar", x="渠道", y="销售额")

    with pytest.raises(ChartError, match="没有可绘制的数值"):
        render_chart(spec, frame)


def test_render_chart_rejects_negative_pie() -> None:
    frame = pd.DataFrame({"渠道": ["线上", "线下"], "销售额": [100, -20]})
    spec = ChartSpec(type="pie", x="渠道", y="销售额")

    with pytest.raises(ChartError, match="非负"):
        render_chart(spec, frame)


def test_render_chart_rejects_missing_column(frame: pd.DataFrame) -> None:
    spec = ChartSpec(type="bar", x="渠道", y="缺失")

    with pytest.raises(ChartError, match="缺少图表列"):
        render_chart(spec, frame)
