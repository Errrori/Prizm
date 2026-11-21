"""澄清门禁轻量评测：复用任务集，统计异常场景召回与整体触发准确率。

用法：
    uv run python -m eval.run_clarify
    uv run python -m eval.run_clarify --tasks eval/tasks.jsonl --datasets-dir eval/datasets

默认走规则/启发式判定，不调用真实模型；目标：
- 异常场景召回 100%
- 整体触发准确率（该问就问 / 不该问不问）≥ 85%
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from eval.metrics import TARGET_CLARIFICATION_ACCURACY, EvalTask
from eval.run_eval import (
    DEFAULT_DATASETS_DIR,
    DEFAULT_TASKS_PATH,
    load_tasks,
    missing_datasets,
)

from prizm.agent.clarify import ClarifyGate
from prizm.datasource.csv_connector import CsvConnector
from prizm.profiling.models import ProfileReport
from prizm.profiling.profiler import profile_frame


def evaluate(tasks: Sequence[EvalTask], datasets_dir: Path) -> list[tuple[EvalTask, bool]]:
    profiles: dict[str, ProfileReport] = {}
    results: list[tuple[EvalTask, bool]] = []
    for task in tasks:
        profile = profiles.get(task.dataset)
        if profile is None:
            frame = CsvConnector(datasets_dir / task.dataset).load()
            profile = profile_frame(frame)
            profiles[task.dataset] = profile
        decision = ClarifyGate().judge(task.question, profile)
        results.append((task, decision.needs_clarification))
    return results


def format_report(results: Sequence[tuple[EvalTask, bool]]) -> tuple[str, bool]:
    expected = [(task, triggered) for task, triggered in results if task.expects_clarification]
    recalled = sum(triggered for _, triggered in expected)
    matched = sum(task.expects_clarification == triggered for task, triggered in results)
    total = len(results)

    recall = recalled / len(expected) if expected else 1.0
    accuracy = matched / total if total else 1.0
    recall_ok = recalled == len(expected)
    accuracy_ok = accuracy >= TARGET_CLARIFICATION_ACCURACY

    lines = [
        "澄清门禁评测",
        f"问题数: {total}（应澄清 {len(expected)}）",
        f"异常场景召回: {recalled}/{len(expected)} = {recall:.1%}（目标 100%）",
        (
            f"整体触发准确率: {matched}/{total} = {accuracy:.1%}"
            f"（目标 {TARGET_CLARIFICATION_ACCURACY:.0%}）"
        ),
    ]
    mismatches = [
        f"- {task.id}: 期望{'澄清' if task.expects_clarification else '直接执行'}，"
        f"实际{'澄清' if triggered else '直接执行'}"
        for task, triggered in results
        if task.expects_clarification != triggered
    ]
    if mismatches:
        lines.append("不一致：")
        lines.extend(mismatches)
    return "\n".join(lines), recall_ok and accuracy_ok


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prizm 澄清门禁轻量评测")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS_PATH)
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
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

    report, passed = format_report(evaluate(tasks, args.datasets_dir))
    print(report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
