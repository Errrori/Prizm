import json
from pathlib import Path
from typing import Any

import pytest
from eval.run_rag import (
    DEFAULT_RAG_TASKS_PATH,
    RagOutcome,
    RagTask,
    format_report,
    load_rag_tasks,
    main,
    parse_rag_task,
)


def test_repository_rag_tasks_are_valid() -> None:
    root = Path(__file__).resolve().parents[2]

    tasks = load_rag_tasks(root / DEFAULT_RAG_TASKS_PATH)

    assert len(tasks) == 50
    assert tasks[0].id == "rag-001"
    assert tasks[-1].id == "rag-050"
    assert {task.dataset for task in tasks} == {"orders.csv"}


def test_main_meets_rag_citation_target(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "50/50 = 100.0%" in output
    assert "目标 90%" in output


def test_format_report_fails_on_mismatch() -> None:
    task = RagTask(
        id="rag-bad",
        dataset="orders.csv",
        question="sales 是什么意思？",
        expected_columns=("sales",),
    )
    outcomes = [
        RagOutcome(task=task, routed=True, correct=False, top_column="channel")
    ]

    report, passed = format_report(outcomes)

    assert passed is False
    assert "不一致" in report
    assert "rag-bad" in report
    assert "期望字段=sales" in report


def test_parse_rag_task_rejects_invalid_fields() -> None:
    with pytest.raises(ValueError, match="expected_columns"):
        parse_rag_task(
            {
                "id": "t1",
                "dataset": "orders.csv",
                "question": "sales 是什么？",
                "expected_columns": [],
            }
        )
    with pytest.raises(ValueError, match="question"):
        parse_rag_task(
            {
                "id": "t1",
                "dataset": "orders.csv",
                "question": " ",
                "expected_columns": ["sales"],
            }
        )


def test_load_rag_tasks_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "rag.jsonl"
    record: dict[str, Any] = {
        "id": "t1",
        "dataset": "orders.csv",
        "question": "sales 是什么？",
        "expected_columns": ["sales"],
    }
    path.write_text(
        json.dumps(record, ensure_ascii=False) + "\n" + json.dumps(record, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="重复 RAG 任务 id"):
        load_rag_tasks(path)
