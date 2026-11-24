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
from dataclasses import asdict, dataclass
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
    anomaly_detection_rate,
    parse_task,
    summarize,
)

from prizm.agent.orchestrator import Answer, Clarification, Orchestrator
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
from prizm.profiling.profiler import profile_frame
from prizm.rag.retriever import DictionaryAnswer
from prizm.utils.tokens import DailyUsageLedger

DEFAULT_TASKS_PATH = Path("eval/tasks.jsonl")
DEFAULT_DATASETS_DIR = Path("eval/datasets")
DEFAULT_OUTCOMES_PATH = Path("eval/outcomes/agent.jsonl")
DEFAULT_ANOMALY_EXPECTATIONS = Path("eval/anomaly_expectations.json")

METRIC_LABELS = {
    "first_pass_success_rate": "一次执行成功率",
    "final_success_rate": "自修复后总成功率",
    "anomaly_detection_rate": "数据异常检出率",
    "clarification_accuracy": "澄清触发正确率",
    "rag_citation_accuracy": "RAG 引用准确率",
}


@dataclass(frozen=True, slots=True)
class TaskDiagnostic:
    """逐题诊断快照：期望值、实际 SQL 与结果摘要，仅用于本地失败归因。"""

    task_id: str
    category: str
    dataset: str
    question: str
    expected: Any
    result_kind: str
    sql: str | None
    columns: list[str]
    rows: list[dict[str, Any]]
    attempts: int
    error: str
    matched: bool | None
    session: str | None = None


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
    _validate_sessions(tasks, path)
    return tasks


def select_tasks(tasks: Sequence[EvalTask], ids: str | Sequence[str]) -> list[EvalTask]:
    """按 id 选中任务；同 session 的任务整体带上，保证多轮上下文可复现。"""
    requested = (
        [item.strip() for item in ids.split(",")]
        if isinstance(ids, str)
        else [item.strip() for item in ids]
    )
    requested = [item for item in requested if item]
    if not requested:
        raise ValueError("--ids 不能为空")
    by_id = {task.id: task for task in tasks}
    unknown = [item for item in requested if item not in by_id]
    if unknown:
        raise ValueError(f"未知任务 id: {', '.join(unknown)}")
    wanted = set(requested)
    sessions = {by_id[item].session for item in requested if by_id[item].session}
    return [
        task
        for task in tasks
        if task.id in wanted or (task.session is not None and task.session in sessions)
    ]


def _validate_sessions(tasks: Sequence[EvalTask], path: Path) -> None:
    """同一 session 的任务必须连续且使用同一数据集，保证串接上下文可复现。"""
    current: str | None = None
    seen: dict[str, str] = {}
    for task in tasks:
        if task.session is None:
            current = None
            continue
        if task.session != current:
            if task.session in seen:
                raise ValueError(
                    f"{path}: 会话 {task.session!r} 的任务必须连续排列"
                )
            seen[task.session] = task.dataset
            current = task.session
        elif seen[task.session] != task.dataset:
            raise ValueError(
                f"{path}: 会话 {task.session!r} 的任务必须使用同一数据集"
            )


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


def _jsonable(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _rows_for_diagnostics(
    frame: pd.DataFrame, *, limit: int = 20
) -> list[dict[str, Any]]:
    return [
        {str(key): _jsonable(value) for key, value in row.items()}
        for row in frame.head(limit).to_dict(orient="records")
    ]


def write_diagnostics(path: Path, diagnostics: Sequence[TaskDiagnostic]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for diagnostic in diagnostics:
            file.write(json.dumps(asdict(diagnostic), ensure_ascii=False) + "\n")


_EXECUTION_ERRORS = (
    LLMError,
    RepairFailed,
    QueryTimeoutError,
    ValueError,
    RuntimeError,
    OSError,
    duckdb.Error,
)


def _build_orchestrator(
    task: EvalTask,
    datasets_dir: Path,
    llm: LLMAdapter,
    engine: DuckDBEngine,
    *,
    max_attempts: int,
) -> Orchestrator:
    connector = CsvConnector(datasets_dir / task.dataset)
    planner = Planner(llm)
    return Orchestrator(
        connector,
        engine,
        planner,
        repairer=Repairer(planner, engine, max_attempts=max_attempts),
    )


def _error_outcome(task: EvalTask, exc: Exception) -> TaskOutcome:
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


def _emit_diagnostic(
    sink: Callable[[TaskDiagnostic], None] | None,
    diagnostic: TaskDiagnostic,
) -> None:
    if sink is not None:
        sink(diagnostic)


def _answer_diagnostic(
    task: EvalTask,
    result: Answer,
    *,
    matched: bool,
    error: str,
) -> TaskDiagnostic:
    return TaskDiagnostic(
        task_id=task.id,
        category=task.category,
        dataset=task.dataset,
        question=task.question,
        expected=task.expected,
        result_kind="answer",
        sql=result.sql,
        columns=[str(column) for column in result.dataframe.columns],
        rows=_rows_for_diagnostics(result.dataframe),
        attempts=result.attempts,
        error=error,
        matched=matched,
        session=task.session,
    )


def _error_diagnostic(task: EvalTask, exc: Exception) -> TaskDiagnostic:
    sql = None
    if isinstance(exc, RepairFailed) and exc.history:
        sql = exc.history[-1].sql
    return TaskDiagnostic(
        task_id=task.id,
        category=task.category,
        dataset=task.dataset,
        question=task.question,
        expected=task.expected,
        result_kind="error",
        sql=sql,
        columns=[],
        rows=[],
        attempts=exc.attempts if isinstance(exc, RepairFailed) else 1,
        error=format_repair_error(exc),
        matched=False,
        session=task.session,
    )


def _run_turn(
    orchestrator: Orchestrator,
    task: EvalTask,
    *,
    on_diagnostic: Callable[[TaskDiagnostic], None] | None = None,
) -> TaskOutcome:
    try:
        result = orchestrator.ask(task.question)
    except _EXECUTION_ERRORS as exc:
        _emit_diagnostic(on_diagnostic, _error_diagnostic(task, exc))
        return _error_outcome(task, exc)

    if isinstance(result, Clarification):
        passed = task.expects_clarification
        error = "" if passed else "未预期的澄清"
        _emit_diagnostic(
            on_diagnostic,
            TaskDiagnostic(
                task_id=task.id,
                category=task.category,
                dataset=task.dataset,
                question=task.question,
                expected=task.expected,
                result_kind="clarification",
                sql=None,
                columns=[],
                rows=[],
                attempts=1,
                error=error,
                matched=passed,
                session=task.session,
            ),
        )
        return TaskOutcome(
            task_id=task.id,
            category=task.category,
            first_pass_ok=passed,
            final_ok=passed,
            expected_clarification=task.expects_clarification,
            clarification_triggered=True,
            attempts=1,
            error=error,
        )

    if isinstance(result, DictionaryAnswer):
        _emit_diagnostic(
            on_diagnostic,
            TaskDiagnostic(
                task_id=task.id,
                category=task.category,
                dataset=task.dataset,
                question=task.question,
                expected=task.expected,
                result_kind="dictionary",
                sql=None,
                columns=[],
                rows=[],
                attempts=1,
                error="误走数据字典问答",
                matched=False,
                session=task.session,
            ),
        )
        return TaskOutcome(
            task_id=task.id,
            category=task.category,
            first_pass_ok=False,
            final_ok=False,
            expected_clarification=task.expects_clarification,
            attempts=1,
            error="误走数据字典问答",
        )

    final_ok = matches_expected(result.dataframe, task.expected)
    error = "" if final_ok else "答案与预期不一致"
    _emit_diagnostic(
        on_diagnostic,
        _answer_diagnostic(task, result, matched=final_ok, error=error),
    )
    return TaskOutcome(
        task_id=task.id,
        category=task.category,
        first_pass_ok=final_ok and result.attempts == 1,
        final_ok=final_ok,
        expected_clarification=task.expects_clarification,
        clarification_triggered=False,
        attempts=result.attempts,
        error=error,
    )


def execute_task(
    task: EvalTask,
    datasets_dir: Path,
    llm: LLMAdapter,
    *,
    max_attempts: int = MAX_REPAIR_ATTEMPTS,
    engine_factory: Callable[[], DuckDBEngine] = DuckDBEngine,
    on_diagnostic: Callable[[TaskDiagnostic], None] | None = None,
) -> TaskOutcome:
    """执行单个任务；单题失败只记录到 TaskOutcome，不中断整批评测。"""
    try:
        with engine_factory() as engine:
            orchestrator = _build_orchestrator(
                task,
                datasets_dir,
                llm,
                engine,
                max_attempts=max_attempts,
            )
            return _run_turn(orchestrator, task, on_diagnostic=on_diagnostic)
    except _EXECUTION_ERRORS as exc:
        _emit_diagnostic(on_diagnostic, _error_diagnostic(task, exc))
        return _error_outcome(task, exc)


def execute_tasks(
    tasks: Sequence[EvalTask],
    datasets_dir: Path,
    llm: LLMAdapter,
    *,
    max_attempts: int = MAX_REPAIR_ATTEMPTS,
    engine_factory: Callable[[], DuckDBEngine] = DuckDBEngine,
    on_result: Callable[[TaskOutcome], None] | None = None,
    on_diagnostic: Callable[[TaskDiagnostic], None] | None = None,
) -> list[TaskOutcome]:
    """按文件顺序执行任务；同一 ``session`` 复用编排器以串接多轮上下文。"""
    outcomes: list[TaskOutcome] = []
    sessions: dict[str, tuple[Orchestrator, DuckDBEngine]] = {}
    try:
        for task in tasks:
            if task.session is None:
                outcome = execute_task(
                    task,
                    datasets_dir,
                    llm,
                    max_attempts=max_attempts,
                    engine_factory=engine_factory,
                    on_diagnostic=on_diagnostic,
                )
            else:
                entry = sessions.get(task.session)
                if entry is None:
                    engine = engine_factory()
                    try:
                        orchestrator = _build_orchestrator(
                            task,
                            datasets_dir,
                            llm,
                            engine,
                            max_attempts=max_attempts,
                        )
                    except _EXECUTION_ERRORS as exc:
                        engine.close()
                        outcome = _error_outcome(task, exc)
                        _emit_diagnostic(
                            on_diagnostic, _error_diagnostic(task, exc)
                        )
                    else:
                        sessions[task.session] = (orchestrator, engine)
                        outcome = _run_turn(
                            orchestrator, task, on_diagnostic=on_diagnostic
                        )
                        if outcome.clarification_triggered:
                            sessions.pop(task.session)[1].close()
                else:
                    outcome = _run_turn(
                        entry[0], task, on_diagnostic=on_diagnostic
                    )
                    if outcome.clarification_triggered:
                        sessions.pop(task.session)[1].close()
            outcomes.append(outcome)
            if on_result is not None:
                on_result(outcome)
    finally:
        for _, engine in sessions.values():
            engine.close()
    return outcomes


def load_anomaly_expectations(path: Path) -> dict[str, tuple[str, ...]]:
    """读取每个数据集应被体检检出的异常代码；文件不存在时返回空映射。"""
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: JSON 解析失败: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: 必须是 JSON 对象")

    expectations: dict[str, tuple[str, ...]] = {}
    for dataset, codes in raw.items():
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError(f"{path}: 数据集名必须是非空字符串")
        if not isinstance(codes, list) or not all(
            isinstance(code, str) and code.strip() for code in codes
        ):
            raise ValueError(f"{path}: {dataset} 的异常代码必须是字符串列表")
        expectations[dataset] = tuple(codes)
    return expectations


def evaluate_anomalies(
    tasks: Sequence[EvalTask],
    datasets_dir: Path,
    expectations: dict[str, tuple[str, ...]],
) -> list[TaskOutcome]:
    """对任务引用到的数据集各体检一次，统计预期异常代码的检出情况。"""
    outcomes: list[TaskOutcome] = []
    profiled: set[str] = set()
    for task in tasks:
        dataset = task.dataset
        if dataset not in expectations or dataset in profiled:
            continue
        profiled.add(dataset)
        frame = CsvConnector(datasets_dir / dataset).load()
        profile = profile_frame(frame)
        expected = tuple(dict.fromkeys(expectations[dataset]))
        detected = sum(code in profile.anomaly_codes for code in expected)
        outcomes.append(
            TaskOutcome(
                task_id=f"anomaly:{dataset}",
                category="cleaning",
                first_pass_ok=True,
                final_ok=True,
                expected_anomalies=len(expected),
                detected_anomalies=detected,
            )
        )
    return outcomes


def with_anomaly_metric(
    metrics: dict[str, MetricResult],
    tasks: Sequence[EvalTask],
    datasets_dir: Path,
    expectations: dict[str, tuple[str, ...]],
) -> dict[str, MetricResult]:
    """用数据集级体检结果覆盖指标表中的异常检出率。"""
    anomaly_outcomes = evaluate_anomalies(tasks, datasets_dir, expectations)
    if not anomaly_outcomes:
        return metrics
    merged = dict(metrics)
    merged["anomaly_detection_rate"] = anomaly_detection_rate(anomaly_outcomes)
    return merged


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
    parser.add_argument(
        "--ids",
        default=None,
        help="仅执行指定任务 id（逗号分隔）；多轮任务自动带上同 session 的前置任务",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=None,
        help="--execute 时把逐题 SQL 与实际结果摘要写入 JSONL",
    )
    parser.add_argument(
        "--anomaly-expectations",
        type=Path,
        default=DEFAULT_ANOMALY_EXPECTATIONS,
        help="数据集级异常检出预期 JSON",
    )
    args = parser.parse_args(argv)

    if args.execute and args.results is not None:
        print("错误: --execute 与 --results 不能同时使用", file=sys.stderr)
        return 2
    if args.limit is not None and (not args.execute or args.limit < 1):
        print("错误: --limit 必须与 --execute 一起使用且 >= 1", file=sys.stderr)
        return 2
    if args.ids is not None and not args.execute:
        print("错误: --ids 必须与 --execute 一起使用", file=sys.stderr)
        return 2
    if args.ids is not None and args.limit is not None:
        print("错误: --ids 与 --limit 不能同时使用", file=sys.stderr)
        return 2
    if args.diagnostics is not None and not args.execute:
        print("错误: --diagnostics 必须与 --execute 一起使用", file=sys.stderr)
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

    missing = missing_datasets(tasks, args.datasets_dir)
    if missing:
        print(f"错误: 缺少数据集: {', '.join(missing)}", file=sys.stderr)
        return 2

    try:
        expectations = load_anomaly_expectations(args.anomaly_expectations)
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    all_tasks = tasks
    if args.execute and args.ids is not None:
        try:
            tasks = select_tasks(tasks, args.ids)
        except ValueError as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return 2
        print(f"已选择任务: {len(tasks)}（含同会话前置）")
    elif args.execute and args.limit is not None:
        tasks = tasks[: args.limit]

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

        diagnostics: list[TaskDiagnostic] = []

        def collect(diagnostic: TaskDiagnostic) -> None:
            diagnostics.append(diagnostic)

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
            on_diagnostic=collect if args.diagnostics is not None else None,
        )
        try:
            write_outcomes(args.outcomes, outcomes)
            if args.diagnostics is not None:
                write_diagnostics(args.diagnostics, diagnostics)
        except OSError as exc:
            print(f"错误: 写入结果失败: {exc}", file=sys.stderr)
            return 2
        print(f"结果已写入: {args.outcomes}")
        if args.diagnostics is not None:
            print(f"诊断已写入: {args.diagnostics}")

        metrics = with_anomaly_metric(
            summarize(outcomes), all_tasks, args.datasets_dir, expectations
        )
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

    metrics = with_anomaly_metric(
        summarize(outcomes), all_tasks, args.datasets_dir, expectations
    )
    print()
    print(format_metric_table(metrics))
    return 0 if all(metric.passed is not False for metric in metrics.values()) else 1


def _as_mapping(raw: Any, path: Path, line_no: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{path}:{line_no}: 每行必须是 JSON 对象")
    return raw


if __name__ == "__main__":
    raise SystemExit(main())
