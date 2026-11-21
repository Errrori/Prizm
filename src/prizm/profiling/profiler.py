"""数据体检：生成紧凑、可注入 prompt 的体检报告。"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any

import pandas as pd

from prizm.profiling.checks import (
    build_duplicate_report,
    coerce_numeric,
    count_missing,
    detect_dirty_numeric,
    detect_missing_anomalies,
    infer_primary_key,
    infer_semantic_type,
)
from prizm.profiling.models import (
    Anomaly,
    ColumnProfile,
    DuplicateReport,
    NumericStats,
    ProfileReport,
)


@dataclass(frozen=True, slots=True)
class ProfileOptions:
    sample_rows: int = 5
    top_values: int = 5
    max_examples: int = 5
    dirty_ratio_threshold: float = 0.8
    max_prompt_chars: int = 2000

    def __post_init__(self) -> None:
        if self.sample_rows < 0:
            raise ValueError("sample_rows 不能为负数")
        if self.top_values < 0:
            raise ValueError("top_values 不能为负数")
        if self.max_examples < 0:
            raise ValueError("max_examples 不能为负数")
        if not 0 < self.dirty_ratio_threshold <= 1:
            raise ValueError("dirty_ratio_threshold 必须在 (0, 1] 内")
        if self.max_prompt_chars < 200:
            raise ValueError("max_prompt_chars 不能小于 200")


def profile_frame(
    frame: pd.DataFrame,
    *,
    primary_key: str | Sequence[str] | None = None,
    options: ProfileOptions | None = None,
) -> ProfileReport:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame 必须是 pandas.DataFrame")
    options = options or ProfileOptions()
    primary_keys = _resolve_primary_key(frame, primary_key)

    columns = tuple(_profile_column(frame[column], options) for column in frame.columns)
    duplicates = build_duplicate_report(
        frame,
        primary_keys,
        max_examples=options.max_examples,
    )
    anomalies = _build_anomalies(frame, duplicates, primary_keys, options)
    sample_rows = _sample_rows(frame, options.sample_rows)
    return ProfileReport(
        row_count=len(frame),
        column_count=len(frame.columns),
        columns=columns,
        duplicates=duplicates,
        anomalies=tuple(anomalies),
        sample_rows=sample_rows,
    )


def format_profile_report(
    report: ProfileReport,
    *,
    max_columns: int = 20,
    include_sample: bool = True,
) -> str:
    lines = [f"数据体检：{report.row_count} 行 × {report.column_count} 列"]
    if report.duplicates.key_columns:
        lines.append(f"疑似主键：{', '.join(report.duplicates.key_columns)}")
    if report.anomalies:
        lines.append(f"异常（{len(report.anomalies)} 项）：")
        lines.extend(f"- {anomaly.message}" for anomaly in report.anomalies)
    else:
        lines.append("异常：未发现")

    if report.columns:
        lines.append(f"列信息（{report.column_count} 列）：")
        lines.extend(_format_column(column) for column in report.columns[:max_columns])
        if report.column_count > max_columns:
            lines.append(f"… 其余 {report.column_count - max_columns} 列已省略")

    if include_sample and report.sample_rows:
        lines.append(f"样本（前 {len(report.sample_rows)} 行）：")
        lines.append(json.dumps(list(report.sample_rows), ensure_ascii=False, default=str))
    return "\n".join(lines)


def format_profile_prompt(report: ProfileReport, *, max_chars: int = 2000) -> str:
    sections = [f"数据体检：{report.row_count} 行 × {report.column_count} 列"]
    if report.duplicates.key_columns:
        sections.append(f"疑似主键：{', '.join(report.duplicates.key_columns)}")
    if report.anomalies:
        sections.append("异常：")
        sections.extend(f"- {anomaly.message}" for anomaly in report.anomalies)
    else:
        sections.append("异常：未发现")

    if report.columns:
        sections.append("列：")
        sections.extend(_format_column(column) for column in report.columns)
    without_sample = "\n".join(sections)

    if report.sample_rows:
        sample = "\n".join(
            [
                f"样本（前 {len(report.sample_rows)} 行，JSON）：",
                json.dumps(list(report.sample_rows), ensure_ascii=False, default=str),
            ]
        )
        text = f"{without_sample}\n{sample}"
    else:
        text = without_sample

    if len(text) <= max_chars:
        return text
    note = "\n（样本因长度限制已省略）"
    if len(without_sample) <= max_chars - len(note):
        return f"{without_sample}{note}"

    suffix = "\n…（体检摘要已截断）"
    return without_sample[: max_chars - len(suffix)].rstrip() + suffix


def _resolve_primary_key(
    frame: pd.DataFrame,
    primary_key: str | Sequence[str] | None,
) -> tuple[str, ...]:
    if primary_key is None:
        inferred = infer_primary_key(frame)
        return (inferred,) if inferred is not None else ()
    keys = (primary_key,) if isinstance(primary_key, str) else tuple(primary_key)
    missing = [key for key in keys if key not in frame.columns]
    if missing:
        raise ValueError(f"主键列不存在: {', '.join(missing)}")
    return tuple(keys)


def _profile_column(series: pd.Series, options: ProfileOptions) -> ColumnProfile:
    semantic_type = infer_semantic_type(series)
    missing_count = count_missing(series)
    total = len(series)
    top_values: tuple[tuple[str, int], ...] = ()
    if semantic_type in {"boolean", "categorical", "text"}:
        top_values = _top_values(series, options.top_values)
    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        semantic_type=semantic_type,
        missing_count=missing_count,
        missing_ratio=missing_count / total if total else 0.0,
        unique_count=_unique_count(series),
        top_values=top_values,
        numeric_stats=_numeric_stats(series) if semantic_type == "numeric" else None,
    )


def _build_anomalies(
    frame: pd.DataFrame,
    duplicates: DuplicateReport,
    primary_keys: tuple[str, ...],
    options: ProfileOptions,
) -> list[Anomaly]:
    anomalies = detect_missing_anomalies(frame)
    if duplicates.exact_duplicate_rows:
        anomalies.append(
            Anomaly(
                code="exact_duplicate_rows",
                severity="warning",
                count=duplicates.exact_duplicate_rows,
                message=f"存在 {duplicates.exact_duplicate_rows} 行完全重复",
            )
        )
    if duplicates.key_duplicate_group_count:
        examples = "；".join(
            f"{'/'.join(primary_keys)}={','.join(group.key_values)}（{group.row_count} 行）"
            for group in duplicates.key_duplicate_examples
        )
        anomalies.append(
            Anomaly(
                code="duplicate_primary_key",
                severity="warning",
                column="/".join(primary_keys),
                count=duplicates.key_duplicate_row_count,
                message=(
                    f"疑似主键 {'/'.join(primary_keys)} 重复："
                    f"{duplicates.key_duplicate_group_count} 组 / "
                    f"{duplicates.key_duplicate_row_count} 行（示例 {examples}）"
                ),
            )
        )
    anomalies.extend(
        detect_dirty_numeric(frame, ratio_threshold=options.dirty_ratio_threshold)
    )
    return anomalies


def _numeric_stats(series: pd.Series) -> NumericStats | None:
    values = coerce_numeric(series).dropna()
    if values.empty:
        return None
    std = values.std()
    return NumericStats(
        min=float(values.min()),
        max=float(values.max()),
        mean=float(values.mean()),
        median=float(values.median()),
        std=None if pd.isna(std) else float(std),
    )


def _unique_count(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return len({_json_value(value) for value in series.dropna()})


def _top_values(series: pd.Series, limit: int) -> tuple[tuple[str, int], ...]:
    if limit <= 0:
        return ()
    text = series.dropna().astype(str).str.strip()
    text = text[text != ""]
    if text.empty:
        return ()
    counts = text.value_counts().head(limit)
    return tuple((str(value), int(count)) for value, count in counts.items())


def _sample_rows(frame: pd.DataFrame, limit: int) -> tuple[dict[str, Any], ...]:
    if limit <= 0 or frame.empty:
        return ()
    rows = frame.head(limit).to_dict(orient="records")
    return tuple({str(key): _json_value(value) for key, value in row.items()} for row in rows)


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value)
    return text if len(text) <= 80 else f"{text[:77]}..."


def _format_column(column: ColumnProfile) -> str:
    parts = [
        f"{column.name}（{column.semantic_type}，{column.dtype}）",
        f"缺失 {column.missing_count}",
        f"唯一 {column.unique_count}",
    ]
    if column.numeric_stats is not None:
        stats = column.numeric_stats
        parts.append(
            f"min={_number(stats.min)} mean={_number(stats.mean)} "
            f"median={_number(stats.median)} max={_number(stats.max)}"
        )
    elif column.top_values:
        top = ", ".join(f"{value}×{count}" for value, count in column.top_values)
        parts.append(f"Top: {top}")
    return "- " + "，".join(parts)


def _number(value: float) -> str:
    return f"{value:.4g}"
