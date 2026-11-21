"""体检报告的数据模型：列画像、异常、重复与整体报告。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SemanticType = Literal["numeric", "datetime", "boolean", "categorical", "text", "unknown"]
AnomalyCode = Literal[
    "missing_values",
    "exact_duplicate_rows",
    "duplicate_primary_key",
    "dirty_numeric_format",
]
Severity = Literal["info", "warning"]


@dataclass(frozen=True, slots=True)
class NumericStats:
    min: float
    max: float
    mean: float
    median: float
    std: float | None = None


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    name: str
    dtype: str
    semantic_type: SemanticType
    missing_count: int
    missing_ratio: float
    unique_count: int
    top_values: tuple[tuple[str, int], ...] = ()
    numeric_stats: NumericStats | None = None


@dataclass(frozen=True, slots=True)
class Anomaly:
    code: AnomalyCode
    severity: Severity
    message: str
    count: int
    column: str | None = None


@dataclass(frozen=True, slots=True)
class KeyDuplicateGroup:
    key_values: tuple[str, ...]
    row_count: int


@dataclass(frozen=True, slots=True)
class DuplicateReport:
    exact_duplicate_rows: int = 0
    key_columns: tuple[str, ...] = ()
    key_duplicate_group_count: int = 0
    key_duplicate_row_count: int = 0
    key_duplicate_examples: tuple[KeyDuplicateGroup, ...] = ()


@dataclass(frozen=True, slots=True)
class ProfileReport:
    row_count: int
    column_count: int
    columns: tuple[ColumnProfile, ...]
    duplicates: DuplicateReport
    anomalies: tuple[Anomaly, ...]
    sample_rows: tuple[dict[str, Any], ...]

    @property
    def has_anomalies(self) -> bool:
        return bool(self.anomalies)

    @property
    def anomaly_codes(self) -> frozenset[str]:
        return frozenset(anomaly.code for anomaly in self.anomalies)

    def to_prompt(self, *, max_chars: int = 2000) -> str:
        from prizm.profiling.profiler import format_profile_prompt

        return format_profile_prompt(self, max_chars=max_chars)
