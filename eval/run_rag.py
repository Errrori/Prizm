"""RAG 术语评测：验证术语问法能被路由到数据字典，并统计引用溯源准确率。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.run_eval import DEFAULT_DATASETS_DIR

from prizm.datasource.csv_connector import CsvConnector
from prizm.profiling.profiler import profile_frame
from prizm.rag.ingest import build_data_dictionary
from prizm.rag.retriever import (
    DataDictionaryRetriever,
    DictionaryQA,
    is_dictionary_question,
)

DEFAULT_RAG_TASKS_PATH = Path("eval/rag_tasks.jsonl")
TARGET_RAG_CITATION_ACCURACY = 0.90


@dataclass(frozen=True, slots=True)
class RagTask:
    id: str
    dataset: str
    question: str
    expected_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RagOutcome:
    task: RagTask
    routed: bool
    correct: bool
    top_column: str | None


def parse_rag_task(raw: Mapping[str, Any]) -> RagTask:
    for field_name in ("id", "dataset", "question", "expected_columns"):
        if field_name not in raw:
            raise ValueError(f"RAG 任务缺少字段: {field_name}")

    task_id = _require_non_empty_str(raw["id"], "id")
    dataset = _require_non_empty_str(raw["dataset"], "dataset")
    question = _require_non_empty_str(raw["question"], "question")
    expected = raw["expected_columns"]
    if not isinstance(expected, list) or not expected:
        raise ValueError("expected_columns 必须是非空列表")
    columns = tuple(_require_non_empty_str(item, "expected_columns") for item in expected)
    return RagTask(
        id=task_id,
        dataset=dataset,
        question=question,
        expected_columns=columns,
    )


def load_rag_tasks(path: Path) -> list[RagTask]:
    tasks: list[RagTask] = []
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
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_no}: 每行必须是 JSON 对象")
            task = parse_rag_task(raw)
            if task.id in seen:
                raise ValueError(f"{path}:{line_no}: 重复 RAG 任务 id: {task.id}")
            seen.add(task.id)
            tasks.append(task)
    if not tasks:
        raise ValueError(f"{path}: 没有 RAG 评测任务")
    return tasks


def missing_datasets(tasks: Sequence[RagTask], datasets_dir: Path) -> list[str]:
    return sorted(
        {
            task.dataset
            for task in tasks
            if not (datasets_dir / task.dataset).is_file()
        }
    )


def build_dictionary_qa(
    tasks: Sequence[RagTask],
    datasets_dir: Path,
) -> dict[str, DictionaryQA]:
    """按数据集构建数据字典问答；数据集只加载一次。"""
    qa_by_dataset: dict[str, DictionaryQA] = {}
    for dataset in sorted({task.dataset for task in tasks}):
        connector = CsvConnector(datasets_dir / dataset)
        frame = connector.load()
        entries = build_data_dictionary(
            frame,
            profile=profile_frame(frame),
            source=dataset,
        )
        qa_by_dataset[dataset] = DictionaryQA(DataDictionaryRetriever(entries))
    return qa_by_dataset


def evaluate(
    tasks: Sequence[RagTask],
    qa_by_dataset: Mapping[str, DictionaryQA],
) -> list[RagOutcome]:
    outcomes: list[RagOutcome] = []
    for task in tasks:
        routed = is_dictionary_question(task.question)
        answer = qa_by_dataset[task.dataset].answer(task.question)
        top_column = answer.citations[0].column if answer.citations else None
        correct = routed and top_column in task.expected_columns
        outcomes.append(
            RagOutcome(
                task=task,
                routed=routed,
                correct=correct,
                top_column=top_column,
            )
        )
    return outcomes


def format_report(outcomes: Sequence[RagOutcome]) -> tuple[str, bool]:
    total = len(outcomes)
    correct = sum(outcome.correct for outcome in outcomes)
    accuracy = correct / total if total else 0.0
    passed = accuracy >= TARGET_RAG_CITATION_ACCURACY

    lines = [
        "RAG 术语问答评测",
        f"问题数: {total}",
        (
            f"引用溯源准确率: {correct}/{total} = {accuracy:.1%}"
            f"（目标 {TARGET_RAG_CITATION_ACCURACY:.0%}）"
        ),
    ]
    mismatches = [
        (
            f"- {outcome.task.id}: 路由={'是' if outcome.routed else '否'}，"
            f"命中字段={outcome.top_column or '无'}，"
            f"期望字段={', '.join(outcome.task.expected_columns)}"
        )
        for outcome in outcomes
        if not outcome.correct
    ]
    if mismatches:
        lines.append("不一致：")
        lines.extend(mismatches)
    return "\n".join(lines), passed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prizm RAG 术语问答评测")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_RAG_TASKS_PATH)
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    args = parser.parse_args(argv)

    try:
        tasks = load_rag_tasks(args.tasks)
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    missing = missing_datasets(tasks, args.datasets_dir)
    if missing:
        print(f"错误: 缺少数据集: {', '.join(missing)}", file=sys.stderr)
        return 2

    # DEBT-019: RAG 评测集为模板生成, 需补充真实业务同义问法。
    qa_by_dataset = build_dictionary_qa(tasks, args.datasets_dir)
    report, passed = format_report(evaluate(tasks, qa_by_dataset))
    print(report)
    return 0 if passed else 1


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 必须是非空字符串")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
