"""数据字典摄取：从 DataFrame 生成列字典，或解析用户上传的说明文档。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from prizm.profiling.models import ColumnProfile, ProfileReport

_DEFAULT_MAX_CHUNK_CHARS = 800
_CANDIDATE_ENCODINGS = ("utf-8-sig", "gbk")
_SEMANTIC_LABELS = {
    "numeric": "数值型",
    "datetime": "日期时间型",
    "boolean": "布尔型",
    "categorical": "类别型",
    "text": "文本型",
    "unknown": "未知类型",
}


@dataclass(frozen=True, slots=True)
class DictionaryEntry:
    id: str
    text: str
    source: str
    column: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("id 不能为空")
        if not self.text.strip():
            raise ValueError("text 不能为空")
        if not self.source.strip():
            raise ValueError("source 不能为空")
        if self.column is not None and not self.column.strip():
            raise ValueError("column 不能为空字符串")


def _format_value(value: Any, *, max_chars: int = 60) -> str:
    try:
        if pd.isna(value):
            return "空"
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text


def _sample_values(series: pd.Series, *, limit: int = 5) -> tuple[str, ...]:
    samples: list[str] = []
    seen: set[str] = set()
    for value in series:
        text = _format_value(value)
        if text == "空" or text in seen:
            continue
        seen.add(text)
        samples.append(text)
        if len(samples) >= limit:
            break
    return tuple(samples)


def _numeric_summary(series: pd.Series) -> str | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return (
        f"数值范围 {numeric.min():g} ~ {numeric.max():g}，"
        f"均值 {numeric.mean():.2f}，中位数 {numeric.median():g}"
    )


def _column_text(
    name: str,
    series: pd.Series,
    column_profile: ColumnProfile | None,
    anomaly_messages: tuple[str, ...],
) -> str:
    dtype = column_profile.dtype if column_profile is not None else str(series.dtype)
    semantic = (
        _SEMANTIC_LABELS.get(column_profile.semantic_type, column_profile.semantic_type)
        if column_profile is not None
        else "未知类型"
    )
    missing = (
        column_profile.missing_count
        if column_profile is not None
        else int(series.isna().sum())
    )
    unique = (
        column_profile.unique_count
        if column_profile is not None
        else int(series.nunique(dropna=True))
    )
    parts = [
        f"字段 {name}（{semantic}，dtype {dtype}）："
        f"共 {len(series)} 行，缺失 {missing}，唯一值 {unique}。"
    ]

    if column_profile is not None and column_profile.top_values:
        top = "、".join(f"{value}({count})" for value, count in column_profile.top_values)
        parts.append(f"高频值：{top}。")

    numeric = None
    if column_profile is not None and column_profile.numeric_stats is not None:
        stats = column_profile.numeric_stats
        numeric = (
            f"数值范围 {stats.min:g} ~ {stats.max:g}，"
            f"均值 {stats.mean:.2f}，中位数 {stats.median:g}"
        )
    elif column_profile is None:
        numeric = _numeric_summary(series)
    if numeric:
        parts.append(f"{numeric}。")

    samples = _sample_values(series)
    if samples:
        parts.append(f"示例值：{'、'.join(samples)}。")
    if anomaly_messages:
        parts.append(f"异常提示：{'；'.join(anomaly_messages)}。")
    return " ".join(parts)


def build_data_dictionary(
    frame: pd.DataFrame,
    *,
    profile: ProfileReport | None = None,
    source: str = "dataset",
) -> tuple[DictionaryEntry, ...]:
    """生成“数据集概览 + 每列说明”的数据字典。"""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame 必须是 pandas.DataFrame")
    if not source.strip():
        raise ValueError("source 不能为空")

    profile_columns = (
        {column.name: column for column in profile.columns} if profile is not None else {}
    )
    anomalies: dict[str, list[str]] = {}
    if profile is not None:
        for anomaly in profile.anomalies:
            if anomaly.column:
                anomalies.setdefault(anomaly.column, []).append(anomaly.message)

    overview_parts = [
        f"数据集 {source}：{len(frame)} 行 × {len(frame.columns)} 列。",
        f"列：{', '.join(str(column) for column in frame.columns)}。",
    ]
    if profile is not None and profile.duplicates.key_columns:
        overview_parts.append(f"疑似主键：{', '.join(profile.duplicates.key_columns)}。")
    if profile is not None and profile.anomalies:
        overview_parts.append(f"体检异常 {len(profile.anomalies)} 项。")
    else:
        overview_parts.append("体检异常：未发现。")

    entries = [
        DictionaryEntry(
            id=f"{source}:overview",
            text=" ".join(overview_parts),
            source=source,
            column=None,
        )
    ]
    for position, column in enumerate(frame.columns):
        name = str(column)
        entries.append(
            DictionaryEntry(
                id=f"{source}:column:{name}",
                text=_column_text(
                    name,
                    frame.iloc[:, position],
                    profile_columns.get(name),
                    tuple(anomalies.get(name, ())),
                ),
                source=source,
                column=name,
            )
        )
    return tuple(entries)


def _chunk_lines(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.split("\n"):
        if not line.strip():
            if current:
                current.append("")
            continue
        if size + len(line) > max_chars and current:
            chunk = "\n".join(current).strip()
            if chunk:
                chunks.append(chunk)
            current = []
            size = 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunk = "\n".join(current).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def parse_dictionary_text(
    text: str,
    *,
    source: str,
    max_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
) -> tuple[DictionaryEntry, ...]:
    """把用户说明文档按段落/标题切成带出处的片段，供 BM25 检索。"""
    if not source.strip():
        raise ValueError("source 不能为空")
    if max_chars < 1:
        raise ValueError("max_chars 必须 >= 1")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [
        block.strip()
        for block in re.split(r"\n\s*\n", normalized)
        if block.strip()
    ]
    merged: list[str] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if (
            block.startswith("#")
            and index + 1 < len(blocks)
            and not blocks[index + 1].startswith("#")
        ):
            merged.append(f"{block}\n{blocks[index + 1]}")
            index += 2
        else:
            merged.append(block)
            index += 1

    chunks: list[str] = []
    for block in merged:
        chunks.extend(_chunk_lines(block, max_chars))

    return tuple(
        DictionaryEntry(id=f"{source}#{index}", text=chunk, source=source)
        for index, chunk in enumerate(chunks, 1)
    )


def load_dictionary_file(
    path: str | Path,
    *,
    source: str | None = None,
    encoding: str | None = None,
    max_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
) -> tuple[DictionaryEntry, ...]:
    """读取 UTF-8(BOM) 或 GBK 的 .md/.txt 数据说明文档。"""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"数据说明文档不存在: {file_path}")

    if encoding is not None:
        text = file_path.read_text(encoding=encoding)
    else:
        last_error: UnicodeDecodeError | None = None
        text = ""
        for candidate in _CANDIDATE_ENCODINGS:
            try:
                text = file_path.read_text(encoding=candidate)
                break
            except UnicodeDecodeError as exc:
                last_error = exc
        else:
            tried = ", ".join(_CANDIDATE_ENCODINGS)
            raise ValueError(
                f"无法解码数据说明文档 {file_path}（已尝试编码: {tried}）"
            ) from last_error
    return parse_dictionary_text(
        text,
        source=source or file_path.name,
        max_chars=max_chars,
    )
