"""chart spec 校验与 matplotlib PNG 渲染（Agg，不执行自由绘图代码）。"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from matplotlib import rcParams
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# DEBT-022: 中文字体依赖系统已安装字体, 容器或精简环境可能缺字显示方框。
_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
)
rcParams["font.sans-serif"] = [*_FONT_CANDIDATES, "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

CHART_TYPES = ("bar", "line", "pie", "scatter")
MAX_CHART_POINTS = 50
_MAX_TITLE_CHARS = 120


class ChartError(ValueError):
    """chart spec 非法或结果数据无法渲染。"""


@dataclass(frozen=True, slots=True)
class ChartSpec:
    type: str
    x: str
    y: str
    title: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type, "x": self.x, "y": self.y}
        if self.title:
            payload["title"] = self.title
        return payload


def parse_chart_spec(
    raw: Mapping[str, Any] | None,
    columns: Sequence[Any],
) -> ChartSpec | None:
    """把 planner 的 chart 字段校验成可渲染的 ChartSpec；None 表示不绘图。"""
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ChartError("chart 必须是对象或 null")

    kind = raw.get("type")
    if not isinstance(kind, str) or not kind.strip():
        raise ChartError("chart.type 不能为空")
    kind = kind.strip().lower()
    if kind not in CHART_TYPES:
        raise ChartError(f"chart.type 仅支持 {', '.join(CHART_TYPES)}")

    names = {str(column) for column in columns}
    for label in ("x", "y"):
        value = raw.get(label)
        if not isinstance(value, str) or not value.strip():
            raise ChartError(f"chart.{label} 必须是非空列名")
        if value not in names:
            raise ChartError(f"chart.{label} 不在结果列中: {value}")

    title = raw.get("title")
    if title is not None and not isinstance(title, str):
        raise ChartError("chart.title 必须是字符串或 null")
    clean_title = title.strip() if isinstance(title, str) else None
    if clean_title == "":
        clean_title = None
    if clean_title and len(clean_title) > _MAX_TITLE_CHARS:
        clean_title = clean_title[:_MAX_TITLE_CHARS] + "…"

    return ChartSpec(type=kind, x=str(raw["x"]), y=str(raw["y"]), title=clean_title)


def render_chart(
    spec: ChartSpec,
    frame: pd.DataFrame,
    *,
    max_points: int = MAX_CHART_POINTS,
) -> bytes:
    """按 chart spec 把结果 DataFrame 渲染成 PNG 字节。"""
    for column in (spec.x, spec.y):
        if column not in frame.columns:
            raise ChartError(f"结果缺少图表列: {column}")

    data = frame[[spec.x, spec.y]].dropna()
    if data.empty:
        raise ChartError("图表数据为空")

    if spec.type == "scatter":
        x_values = pd.to_numeric(data[spec.x], errors="coerce")
        y_values = pd.to_numeric(data[spec.y], errors="coerce")
        valid = x_values.notna() & y_values.notna()
        x_values = x_values[valid]
        y_values = y_values[valid]
        labels: pd.Series | None = None
    else:
        y_values = pd.to_numeric(data[spec.y], errors="coerce")
        valid = y_values.notna()
        data = data[valid]
        y_values = y_values[valid]
        labels = data[spec.x].astype(str)
        x_values = labels

    if y_values.empty:
        raise ChartError("图表 y 列没有可绘制的数值")

    if len(y_values) > max_points:
        x_values = x_values.head(max_points)
        y_values = y_values.head(max_points)
        labels = labels.head(max_points) if labels is not None else None

    if spec.type == "pie":
        if (y_values < 0).any():
            raise ChartError("饼图要求 y 列数值非负")
        if float(y_values.sum()) == 0:
            raise ChartError("饼图 y 列合计为 0，无法绘制")

    figure = Figure(figsize=(6.4, 4.0), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)

    if spec.type == "bar":
        axes.bar(x_values.tolist(), y_values.tolist())
        axes.set_xlabel(spec.x)
        axes.set_ylabel(spec.y)
    elif spec.type == "line":
        axes.plot(x_values.tolist(), y_values.tolist(), marker="o")
        axes.set_xlabel(spec.x)
        axes.set_ylabel(spec.y)
    elif spec.type == "pie":
        assert labels is not None
        axes.pie(y_values.tolist(), labels=labels.tolist(), autopct="%1.1f%%")
    else:
        axes.scatter(x_values.tolist(), y_values.tolist())
        axes.set_xlabel(spec.x)
        axes.set_ylabel(spec.y)

    if spec.type in {"bar", "line"} and len(x_values) > 8:
        axes.tick_params(axis="x", labelrotation=45)
        for label in axes.get_xticklabels():
            label.set_horizontalalignment("right")

    axes.set_title(spec.title or f"{spec.y} 按 {spec.x}")
    figure.tight_layout()

    buffer = io.BytesIO()
    canvas.print_png(buffer)
    return buffer.getvalue()
