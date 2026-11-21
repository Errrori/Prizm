"""体检的单项检测：缺失、重复、脏数值格式与主键推断。"""

from __future__ import annotations

import re
from collections.abc import Sequence

import pandas as pd

from prizm.profiling.models import (
    Anomaly,
    DuplicateReport,
    KeyDuplicateGroup,
    SemanticType,
)

_SYMBOL_RE = re.compile(r"[¥$€£]")
_UNIT_RE = re.compile(r"(?:万元|元|万|亿|千|百|%|％)")
_DIRTY_MARKER_RE = re.compile(r"[¥$€£]|\d[,，]\d{3}|万元|元|万|亿|千|百|%|％")
_NUMERIC_RATIO_THRESHOLD = 0.9
_DATETIME_RATIO_THRESHOLD = 0.9


def missing_mask(series: pd.Series) -> pd.Series:
    """NaN、空字符串与纯空白都视为缺失。"""
    mask = series.isna()
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        text = series.astype("string").str.strip()
        mask = mask | text.eq("").fillna(False)
    return mask


def count_missing(series: pd.Series) -> int:
    return int(missing_mask(series).sum())


def detect_missing_anomalies(frame: pd.DataFrame) -> list[Anomaly]:
    anomalies: list[Anomaly] = []
    total = len(frame)
    for column in frame.columns:
        count = count_missing(frame[column])
        if count == 0:
            continue
        ratio = count / total if total else 0.0
        anomalies.append(
            Anomaly(
                code="missing_values",
                severity="warning",
                column=str(column),
                count=count,
                message=f"列 {column} 有 {count} 个缺失值（{ratio:.1%}）",
            )
        )
    return anomalies


def infer_primary_key(frame: pd.DataFrame) -> str | None:
    """按列名启发式识别主键：`id` 或 `*_id`，取第一个匹配列。"""
    # DEBT-007: 主键按列名启发式取第一个匹配列, 可能误判/漏判。
    for column in frame.columns:
        name = str(column).strip().lower()
        if name == "id" or name.endswith("_id"):
            return str(column)
    return None


def build_duplicate_report(
    frame: pd.DataFrame,
    primary_key: Sequence[str] = (),
    *,
    max_examples: int = 5,
) -> DuplicateReport:
    exact = int(frame.duplicated().sum())
    keys = tuple(str(key) for key in primary_key)
    if not keys:
        return DuplicateReport(exact_duplicate_rows=exact)

    missing_keys = [key for key in keys if key not in frame.columns]
    if missing_keys:
        raise ValueError(f"主键列不存在: {', '.join(missing_keys)}")

    scoped = frame.dropna(subset=list(keys))
    if scoped.empty:
        return DuplicateReport(exact_duplicate_rows=exact, key_columns=keys)

    grouped = scoped.groupby(list(keys), dropna=False).size()
    duplicated = grouped[grouped > 1]
    if duplicated.empty:
        return DuplicateReport(exact_duplicate_rows=exact, key_columns=keys)

    examples = tuple(
        KeyDuplicateGroup(
            key_values=tuple(_display_value(part) for part in _as_tuple(index)),
            row_count=int(count),
        )
        for index, count in duplicated.sort_values(ascending=False).head(max_examples).items()
    )
    return DuplicateReport(
        exact_duplicate_rows=exact,
        key_columns=keys,
        key_duplicate_group_count=len(duplicated),
        key_duplicate_row_count=int(duplicated.sum()),
        key_duplicate_examples=examples,
    )


def detect_dirty_numeric(
    frame: pd.DataFrame,
    *,
    ratio_threshold: float = 0.8,
) -> list[Anomaly]:
    """检测对象列中混入千分位/货币符/单位/百分号的数值。"""
    # DEBT-008: 脏数值检测基于千分位/货币/单位/百分号启发式, 非完整类型推断。
    anomalies: list[Anomaly] = []
    for column in frame.columns:
        series = frame[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        text = series.dropna().astype(str).str.strip()
        text = text[text != ""]
        if text.empty:
            continue
        normalized = _normalize_numeric_text(text)
        parsed = pd.to_numeric(normalized, errors="coerce")
        if parsed.notna().mean() < ratio_threshold:
            continue
        dirty_mask = text.str.contains(_DIRTY_MARKER_RE, regex=True, na=False) & parsed.notna()
        dirty_count = int(dirty_mask.sum())
        if dirty_count == 0:
            continue
        examples = ", ".join(repr(value) for value in text[dirty_mask].head(3))
        anomalies.append(
            Anomaly(
                code="dirty_numeric_format",
                severity="warning",
                column=str(column),
                count=dirty_count,
                message=f"列 {column} 有 {dirty_count} 个值混入千分位/单位（示例 {examples}）",
            )
        )
    return anomalies


def infer_semantic_type(series: pd.Series) -> SemanticType:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return "unknown"

    sample = values.head(1000)
    if _numeric_ratio(sample) >= _NUMERIC_RATIO_THRESHOLD:
        return "numeric"
    if _datetime_ratio(sample) >= _DATETIME_RATIO_THRESHOLD:
        return "datetime"
    if sample.nunique() / len(sample) <= 0.5:
        return "categorical"
    return "text"


def _numeric_ratio(values: pd.Series) -> float:
    normalized = _normalize_numeric_text(values)
    return float(pd.to_numeric(normalized, errors="coerce").notna().mean())


def _datetime_ratio(values: pd.Series) -> float:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def coerce_numeric(series: pd.Series) -> pd.Series:
    """把数值列或带千分位/单位装饰的字符串列转成数值，失败为 NaN。"""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip()
    normalized = _normalize_numeric_text(text.fillna(""))
    return pd.to_numeric(normalized, errors="coerce")


def _normalize_numeric_text(values: pd.Series) -> pd.Series:
    cleaned = values.str.replace(",", "", regex=False).str.replace("，", "", regex=False)
    cleaned = cleaned.str.replace(_SYMBOL_RE, "", regex=True)
    cleaned = cleaned.str.replace(_UNIT_RE, "", regex=True)
    return cleaned.str.replace(r"\s+", "", regex=True)


def _as_tuple(index: object) -> tuple[object, ...]:
    if isinstance(index, tuple):
        return index
    return (index,)


def _display_value(value: object) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= 40 else f"{text[:37]}..."
