"""评测入口：校验任务 / 执行 Agent / 读取结果并输出指标。

用法：
    uv run python -m eval.run_eval
    uv run python -m eval.run_eval --execute --limit 3
    uv run python -m eval.run_eval --results eval/outcomes/agent.jsonl

默认只校验任务文件与数据集；``--execute`` 用真实模型逐题执行 Agent
并产出 :class:`eval.metrics.TaskOutcome` JSONL；``--results`` 读取已有
结果 JSONL 并输出指标表。mock 模型无法评测，``--execute`` 会直接拒绝。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import date, datetime
from numbers import Real
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from eval.metrics import (
    CATEGORIES,
    EvalTask,
    MetricResult,
    TaskOutcome,
    parse_task,
    summarize,
)

from prizm.agent.orchestrator import Clarification, Orchestrator
from prizm.agent.planner import Planner
from prizm.agent.repair import (
    MAX_REPAIR_ATTEMPTS,
    Repairer,
    RepairFailed,
    format_repair_error,
)
from prizm.config import get_settings
from prizm.datasource.csv_connector import CsvConnector
from prizm.engine.duckdb_engine import DuckDBEngine, QueryTimeoutError
from prizm.llm.adapter import LLMAdapter, LLMError, build_llm
from prizm.utils.tokens import DailyUsageLedger

DEFAULT_TASKS_PATH = Path("eval/tasks.jsonl")
DEFAULT_DATASETS_DIR = Path("eval/datasets")
DEFAULT_OUTCOMES_PATH = Path("eval/outcomes/agent.jsonl")

METRIC_LABELS = {
    "first_pass_success_rate": "一次执行成功率",
    "final_success_rate": "自修复后总成功率",
    "anomaly_detection_rate": "数据异常检出率",
    "clarification_accuracy": "澄清触发正确率",
    "rag_citation_accuracy": "RAG 引用准确率",
}


def load_tasks(path: Path) -> list[EvalTask]:
    tasks: list[EvalTask] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: JSON 解析失败: {exc.msg}") from exc
            task = parse_task(raw)
            if task.id in seen:
                raise ValueError(f"{path}:{line_no}: 重复任务 id: {task.id}")
            seen.add(task.id)
            tasks.append(task)
    if not tasks:
        raise ValueError(f"{path}: 没有评测任务")
    return tasks


def missing_datasets(tasks: Sequence[EvalTask], datasets_dir: Path) -> list[str]:
    return sorted(
        {
            task.dataset
            for task in tasks
            if not (datasets_dir / task.dataset).is_file()
        }
    )


def load_outcomes(path: Path) -> list[TaskOutcome]:
    outcomes: list[TaskOutcome] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: JSON 解析失败: {exc.msg}") from exc
            try:
                outcome = TaskOutcome(**_as_mapping(raw, path, line_no))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: 结果字段非法: {exc}") from exc
            if outcome.task_id in seen:
                raise ValueError(f"{path}:{line_no}: 重复结果 task_id: {outcome.task_id}")
            seen.add(outcome.task_id)
            outcomes.append(outcome)
    if not outcomes:
        raise ValueError(f"{path}: 没有评测结果")
    return outcomes


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    try:
        return bool(missing)
    except ValueError:
        return False


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _values_equal(expected: Any, actual: Any) -> bool:
    expected_missing = _is_missing(expected)
    actual_missing = _is_missing(actual)
    if expected_missing or actual_missing:
        return expected_missing and actual_missing

    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) and expected == actual

    if isinstance(actual, (datetime, date)) and isinstance(expected, str):
        expected_text = expected.strip()
        candidates = {
            actual.isoformat(),
            actual.strftime("%Y-%m-%d"),
            actual.strftime("%Y-%m"),
            actual.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return expected_text in candidates

    if _is_number(expected) and _is_number(actual):
        return math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-9)
    if _is_number(expected) and isinstance(actual, str):
        try:
            actual_number = float(actual.strip())
        except ValueError:
            return False
        return math.isclose(float(expected), actual_number, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(expected, str):
        return str(actual).strip() == expected.strip()
    return bool(expected == actual)


def _matches_records(frame: pd.DataFrame, expected: Sequence[Any]) -> bool:
    if len(frame) != len(expected):
        return False
    actual_rows = frame.to_dict(orient="records")
    for expected_row, actual_row in zip(expected, actual_rows, strict=True):
        if not isinstance(expected_row, Mapping):
            return False
        for key, expected_value in expected_row.items():
            if key not in actual_row:
                return False
            if not _values_equal(expected_value, actual_row[key]):
                return False
    return True


# DEBT-013: 评测答案比对为启发式, 正式验收前需复核。
def matches_expected(frame: pd.DataFrame, expected: Any) -> bool:
    """把结果 DataFrame 与任务标准答案做启发式比对。

    - 标量答案要求恰好一行一列。
    - 记录列表按顺序逐行比较；只要求期望字段匹配，允许结果多出列。
    """
    if expected is None:
        return False
    if isinstance(expected, list):
        if not expected:
            return bool(frame.empty)
        if isinstance(expected[0], Mapping):
            return _matches_records(frame, expected)
        if frame.shape[1] != 1 or len(frame) != len(expected):
            return False
        return all(
            _values_equal(expected_value, actual_value)
            for expected_value, actual_value in zip(expected, frame.iloc[:, 0], strict=True)
        )
    if frame.shape != (1, 1):
        return False
    return _values_equal(expected, frame.iloc[0, 0])


def write_outcomes(path: Path, outcomes: Sequence[TaskOutcome]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for outcome in outcomes:
            file.write(json.dumps(asdict(outcome), ensure_ascii=False) + "\n")


def execute_task(
    task: EvalTask,
    datasets_dir: Path,
    llm: LLMAdapter,
    *,
    max_attempts: int = MAX_REPAIR_ATTEMPTS,
    engine_factory: Callable[[], DuckDBEngine] = DuckDBEngine,
) -> TaskOutcome:
    """执行单个任务；单题失败只记录到 TaskOutcome，不中断整批评测。"""
    try:
        with engine_factory() as engine:
            connector = CsvConnector(datasets_dir / task.dataset)
            planner = Planner(llm)
            orchestrator = Orchestrator(
                connector,
                engine,
                planner,
                repairer=Repairer(planner, engine, max_attempts=max_attempts),
            )
            result = orchestrator.ask(task.question)
    except (
        LLMError,
        RepairFailed,
        QueryTimeoutError,
        ValueError,
        RuntimeError,
        OSError,
        duckdb.Error,
    ) as exc:
        attempts = exc.attempts if isinstance(exc, RepairFailed) else 1
        return TaskOutcome(
            task_id=task.id,
            category=task.category,
            first_pass_ok=False,
            final_ok=False,
            expected_clarification=task.expects_clarification,
            attempts=attempts,
            error=format_repair_error(exc),
        )

    if isinstance(result, Clarification):
        passed = task.expects_clarification
        return TaskOutcome(
            task_id=task.id,
            category=task.category,
            first_pass_ok=passed,
            final_ok=passed,
            expected_clarification=task.expects_clarification,
            clarification_triggered=True,
            attempts=1,
            error="" if passed else "未预期的澄清",
        )

    final_ok = matches_expected(result.dataframe, task.expected)
    return TaskOutcome(
        task_id=task.id,
        category=task.category,
        first_pass_ok=final_ok and result.attempts == 1,
        final_ok=final_ok,
        expected_clarification=task.expects_clarification,
        clarification_triggered=False,
        attempts=result.attempts,
        error="" if final_ok else "答案与预期不一致",
    )


def execute_tasks(
    tasks: Sequence[EvalTask],
    datasets_dir: Path,
    llm: LLMAdapter,
    *,
    max_attempts: int = MAX_REPAIR_ATTEMPTS,
    engine_factory: Callable[[], DuckDBEngine] = DuckDBEngine,
    on_result: Callable[[TaskOutcome], None] | None = None,
) -> list[TaskOutcome]:
    outcomes: list[TaskOutcome] = []
    for task in tasks:
        outcome = execute_task(
            task,
            datasets_dir,
            llm,
            max_attempts=max_attempts,
            engine_factory=engine_factory,
        )
        outcomes.append(outcome)
        if on_result is not None:
            on_result(outcome)
    return outcomes


def format_metric_table(metrics: dict[str, MetricResult]) -> str:
    lines = [
        f"{'指标':<28} {'值':>8} {'分子/分母':>12} {'目标':>8} {'状态':>6}",
        "-" * 70,
    ]
    for name, metric in metrics.items():
        label = METRIC_LABELS.get(name, name)
        value = "N/A" if metric.value is None else f"{metric.value:.1%}"
        status = "N/A" if metric.passed is None else ("PASS" if metric.passed else "FAIL")
        fraction = f"{metric.numerator}/{metric.denominator}"
        lines.append(
            f"{label:<28} {value:>8} {fraction:>12} {metric.target:>7.0%} {status:>6}"
        )
    return "\n".join(lines)


def format_task_summary(tasks: Sequence[EvalTask]) -> str:
    counts = Counter(task.category for task in tasks)
    lines = [f"任务数: {len(tasks)}"]
    for category in sorted(CATEGORIES):
        lines.append(f"  {category}: {counts.get(category, 0)}")
    clarify_count = sum(task.expects_clarification for task in tasks)
    lines.append(f"需澄清: {clarify_count}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prizm 最小评测入口")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS_PATH)
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--results", type=Path, help="读取已有 TaskOutcome JSONL")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="用真实模型逐题执行 Agent 并产出 TaskOutcome",
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=DEFAULT_OUTCOMES_PATH,
        help="--execute 的结果输出路径",
    )
    parser.add_argument("--max-attempts", type=int, default=MAX_REPAIR_ATTEMPTS)
    parser.add_argument("--limit", type=int, default=None, help="仅执行前 N 题")
    args = parser.parse_args(argv)

    if args.execute and args.results is not None:
        print("错误: --execute 与 --results 不能同时使用", file=sys.stderr)
        return 2
    if args.limit is not None and (not args.execute or args.limit < 1):
        print("错误: --limit 必须与 --execute 一起使用且 >= 1", file=sys.stderr)
        return 2
    if args.execute and not 1 <= args.max_attempts <= MAX_REPAIR_ATTEMPTS:
        print(
            f"错误: --max-attempts 必须在 1 到 {MAX_REPAIR_ATTEMPTS} 之间",
            file=sys.stderr,
        )
        return 2

    try:
        tasks = load_tasks(args.tasks)
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    if args.execute and args.limit is not None:
        tasks = tasks[: args.limit]

    missing = missing_datasets(tasks, args.datasets_dir)
    if missing:
        print(f"错误: 缺少数据集: {', '.join(missing)}", file=sys.stderr)
        return 2

    print(format_task_summary(tasks))

    if args.execute:
        settings = get_settings()
        if settings.is_mock:
            print(
                "错误: --execute 需要真实模型配置（PRIZM_DEFAULT_MODEL 与对应 API key），"
                "mock 模式无法评测。",
                file=sys.stderr,
            )
            return 2
        try:
            llm = build_llm(
                settings,
                ledger=DailyUsageLedger(),
                daily_token_budget=settings.daily_token_budget,
            )
        except LLMError as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return 2

        def report(outcome: TaskOutcome) -> None:
            status = "PASS" if outcome.final_ok else "FAIL"
            print(f"{outcome.task_id}: {status}（尝试 {outcome.attempts} 次）", flush=True)
            if outcome.error:
                print(f"  {outcome.error}", flush=True)

        def engine_factory() -> DuckDBEngine:
            return DuckDBEngine(
                memory_limit=settings.duckdb_memory_limit,
                threads=settings.duckdb_threads,
                timeout_seconds=settings.sql_timeout_seconds,
            )

        outcomes = execute_tasks(
            tasks,
            args.datasets_dir,
            llm,
            max_attempts=args.max_attempts,
            engine_factory=engine_factory,
            on_result=report,
        )
        try:
            write_outcomes(args.outcomes, outcomes)
        except OSError as exc:
            print(f"错误: 写入结果失败: {exc}", file=sys.stderr)
            return 2
        print(f"结果已写入: {args.outcomes}")

        metrics = summarize(outcomes)
        print()
        print(format_metric_table(metrics))
        return 0 if all(metric.passed is not False for metric in metrics.values()) else 1

    if args.results is None:
        print("未提供 --execute 或 --results，仅完成任务与数据集校验。")
        print("用 --execute 跑 Agent，或用 --results 读取已有 TaskOutcome JSONL。")
        return 0

    try:
        outcomes = load_outcomes(args.results)
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    metrics = summarize(outcomes)
    print()
    print(format_metric_table(metrics))
    return 0 if all(metric.passed is not False for metric in metrics.values()) else 1


def _as_mapping(raw: Any, path: Path, line_no: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{path}:{line_no}: 每行必须是 JSON 对象")
    return raw


if __name__ == "__main__":
    raise SystemExit(main())
