"""最小评测入口骨架。

用法：
    uv run python -m eval.run_eval
    uv run python -m eval.run_eval --results path/to/outcomes.jsonl

默认只校验任务文件与数据集；提供 ``--results`` 后，读取逐题
:class:`eval.metrics.TaskOutcome` JSONL 并输出指标表。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.metrics import (
    CATEGORIES,
    EvalTask,
    MetricResult,
    TaskOutcome,
    parse_task,
    summarize,
)

DEFAULT_TASKS_PATH = Path("eval/tasks.jsonl")
DEFAULT_DATASETS_DIR = Path("eval/datasets")

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
    parser.add_argument("--results", type=Path, help="TaskOutcome JSONL 路径")
    args = parser.parse_args(argv)

    try:
        tasks = load_tasks(args.tasks)
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    missing = missing_datasets(tasks, args.datasets_dir)
    if missing:
        print(f"错误: 缺少数据集: {', '.join(missing)}", file=sys.stderr)
        return 2

    print(format_task_summary(tasks))

    if args.results is None:
        print("未提供 --results，仅完成任务与数据集校验。")
        print("接入 Agent 后产出 TaskOutcome JSONL，再用 --results 计算指标。")
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
