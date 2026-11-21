"""评测任务格式与指标计算。

任务文件为 JSONL，每行一个对象，字段如下：

- ``id``：唯一标识，非空字符串。
- ``category``：``cleaning`` / ``aggregation`` / ``trend`` / ``attribution`` / ``multi_turn``。
- ``dataset``：相对 ``eval/datasets`` 的数据文件名。
- ``question``：自然语言问题。
- ``expected``：标准答案，必须是 JSON 可序列化值。
- ``expects_clarification``：是否应触发澄清，默认 ``False``。

``eval/run_eval.py`` 逐题执行后产出 :class:`TaskOutcome`，再交给本模块计算指标。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "cleaning",
        "aggregation",
        "trend",
        "attribution",
        "multi_turn",
    }
)

TARGET_FIRST_PASS_SUCCESS: Final[float] = 0.75
TARGET_FINAL_SUCCESS: Final[float] = 0.90
TARGET_ANOMALY_DETECTION: Final[float] = 1.00
TARGET_CLARIFICATION_ACCURACY: Final[float] = 0.85
TARGET_RAG_CITATION_ACCURACY: Final[float] = 0.90

_REQUIRED_TASK_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "category",
    "dataset",
    "question",
    "expected",
)


@dataclass(frozen=True, slots=True)
class EvalTask:
    id: str
    category: str
    dataset: str
    question: str
    expected: Any
    expects_clarification: bool = False


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    task_id: str
    category: str
    first_pass_ok: bool
    final_ok: bool
    expected_clarification: bool = False
    clarification_triggered: bool = False
    attempts: int = 1
    error: str = ""
    expected_anomalies: int = 0
    detected_anomalies: int = 0
    rag_citations_total: int = 0
    rag_citations_correct: int = 0

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"未知 category: {self.category!r}")
        if (
            not isinstance(self.attempts, int)
            or isinstance(self.attempts, bool)
            or self.attempts < 1
        ):
            raise ValueError("attempts 必须是 >= 1 的整数")
        if not isinstance(self.error, str):
            raise ValueError("error 必须是字符串")
        for field_name in (
            "expected_anomalies",
            "detected_anomalies",
            "rag_citations_total",
            "rag_citations_correct",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} 必须是非负整数")
        if self.detected_anomalies > self.expected_anomalies:
            raise ValueError("detected_anomalies 不能大于 expected_anomalies")
        if self.rag_citations_correct > self.rag_citations_total:
            raise ValueError("rag_citations_correct 不能大于 rag_citations_total")


@dataclass(frozen=True, slots=True)
class MetricResult:
    name: str
    value: float | None
    numerator: int
    denominator: int
    target: float

    @property
    def passed(self) -> bool | None:
        if self.value is None:
            return None
        return self.value >= self.target


def parse_task(raw: Mapping[str, Any]) -> EvalTask:
    missing = [field for field in _REQUIRED_TASK_FIELDS if field not in raw]
    if missing:
        raise ValueError(f"任务缺少字段: {', '.join(missing)}")

    category = _require_non_empty_str(raw["category"], "category")
    if category not in CATEGORIES:
        allowed = ", ".join(sorted(CATEGORIES))
        raise ValueError(f"未知 category: {category!r}，可选值: {allowed}")

    expects_clarification = raw.get("expects_clarification", False)
    if not isinstance(expects_clarification, bool):
        raise ValueError("expects_clarification 必须是 bool")

    return EvalTask(
        id=_require_non_empty_str(raw["id"], "id"),
        category=category,
        dataset=_require_non_empty_str(raw["dataset"], "dataset"),
        question=_require_non_empty_str(raw["question"], "question"),
        expected=raw["expected"],
        expects_clarification=expects_clarification,
    )


def first_pass_success_rate(outcomes: Iterable[TaskOutcome]) -> MetricResult:
    items = tuple(outcomes)
    return _rate(
        "first_pass_success_rate",
        sum(outcome.first_pass_ok for outcome in items),
        len(items),
        TARGET_FIRST_PASS_SUCCESS,
    )


def final_success_rate(outcomes: Iterable[TaskOutcome]) -> MetricResult:
    items = tuple(outcomes)
    return _rate(
        "final_success_rate",
        sum(outcome.final_ok for outcome in items),
        len(items),
        TARGET_FINAL_SUCCESS,
    )


def anomaly_detection_rate(outcomes: Iterable[TaskOutcome]) -> MetricResult:
    items = tuple(outcomes)
    return _rate(
        "anomaly_detection_rate",
        sum(outcome.detected_anomalies for outcome in items),
        sum(outcome.expected_anomalies for outcome in items),
        TARGET_ANOMALY_DETECTION,
    )


def clarification_accuracy(outcomes: Iterable[TaskOutcome]) -> MetricResult:
    items = tuple(outcomes)
    return _rate(
        "clarification_accuracy",
        sum(
            outcome.expected_clarification == outcome.clarification_triggered
            for outcome in items
        ),
        len(items),
        TARGET_CLARIFICATION_ACCURACY,
    )


def rag_citation_accuracy(outcomes: Iterable[TaskOutcome]) -> MetricResult:
    items = tuple(outcomes)
    return _rate(
        "rag_citation_accuracy",
        sum(outcome.rag_citations_correct for outcome in items),
        sum(outcome.rag_citations_total for outcome in items),
        TARGET_RAG_CITATION_ACCURACY,
    )


def summarize(outcomes: Iterable[TaskOutcome]) -> dict[str, MetricResult]:
    items = tuple(outcomes)
    return {
        metric.name: metric
        for metric in (
            first_pass_success_rate(items),
            final_success_rate(items),
            anomaly_detection_rate(items),
            clarification_accuracy(items),
            rag_citation_accuracy(items),
        )
    }


def _rate(name: str, numerator: int, denominator: int, target: float) -> MetricResult:
    value = None if denominator == 0 else numerator / denominator
    return MetricResult(
        name=name,
        value=value,
        numerator=numerator,
        denominator=denominator,
        target=target,
    )


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 必须是非空字符串")
    return value
