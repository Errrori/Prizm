import json
from pathlib import Path
from typing import Any

import pytest
from eval.metrics import EvalTask, TaskOutcome, summarize
from eval.run_eval import (
    evaluate_anomalies,
    format_metric_table,
    format_task_summary,
    load_anomaly_expectations,
    load_outcomes,
    load_tasks,
    main,
    missing_datasets,
    with_anomaly_metric,
)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_task(task_id: str = "task-1", **overrides: Any) -> EvalTask:
    values: dict[str, Any] = {
        "id": task_id,
        "category": "aggregation",
        "dataset": "orders.csv",
        "question": "销售额是多少？",
        "expected": 1,
    }
    values.update(overrides)
    return EvalTask(**values)


def test_load_tasks_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "t1",
                "category": "aggregation",
                "dataset": "orders.csv",
                "question": "销售额是多少？",
                "expected": 100,
            },
            {
                "id": "t2",
                "category": "cleaning",
                "dataset": "orders_dirty.csv",
                "question": "是否应去重？",
                "expected": None,
                "expects_clarification": True,
            },
        ],
    )

    tasks = load_tasks(path)

    assert [task.id for task in tasks] == ["t1", "t2"]
    assert tasks[1].expects_clarification is True


def test_load_tasks_rejects_duplicate_ids(tmp_path: Path) -> None:
    record = {
        "id": "t1",
        "category": "aggregation",
        "dataset": "orders.csv",
        "question": "销售额是多少？",
        "expected": 100,
    }
    path = tmp_path / "tasks.jsonl"
    write_jsonl(path, [record, record])

    with pytest.raises(ValueError, match="重复任务 id: t1"):
        load_tasks(path)


def test_load_tasks_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="没有评测任务"):
        load_tasks(path)


def test_missing_datasets_reports_only_absent_files(tmp_path: Path) -> None:
    (tmp_path / "orders.csv").write_text("order_id\n1\n", encoding="utf-8")
    tasks = [
        make_task("t1", dataset="orders.csv"),
        make_task("t2", dataset="missing.csv"),
    ]

    assert missing_datasets(tasks, tmp_path) == ["missing.csv"]


def test_format_task_summary_counts_categories_and_clarifications() -> None:
    tasks = [
        make_task("t1", category="aggregation"),
        make_task("t2", category="cleaning", expects_clarification=True),
    ]

    output = format_task_summary(tasks)

    assert "任务数: 2" in output
    assert "aggregation: 1" in output
    assert "cleaning: 1" in output
    assert "需澄清: 1" in output


def test_repository_minimal_tasks_are_valid() -> None:
    root = Path(__file__).resolve().parents[2]
    tasks = load_tasks(root / "eval" / "tasks.jsonl")

    assert len(tasks) == 150
    assert sum(task.expects_clarification for task in tasks) == 12
    assert missing_datasets(tasks, root / "eval" / "datasets") == []
    assert {task.category for task in tasks} == {
        "cleaning",
        "aggregation",
        "trend",
        "attribution",
        "multi_turn",
    }
    assert any(task.session for task in tasks)
    assert {task.dataset for task in tasks} == {
        "orders.csv",
        "orders_dirty.csv",
        "monthly_sales.csv",
        "monthly_sales_dirty.csv",
        "channel_traffic.csv",
        "channel_traffic_dirty.csv",
    }


def test_load_tasks_rejects_non_contiguous_sessions(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "t1",
                "category": "multi_turn",
                "dataset": "orders.csv",
                "question": "第一轮",
                "expected": 1,
                "session": "s1",
            },
            {
                "id": "t2",
                "category": "multi_turn",
                "dataset": "orders.csv",
                "question": "独立一轮",
                "expected": 1,
            },
            {
                "id": "t3",
                "category": "multi_turn",
                "dataset": "orders.csv",
                "question": "又回到会话",
                "expected": 1,
                "session": "s1",
            },
        ],
    )

    with pytest.raises(ValueError, match="必须连续排列"):
        load_tasks(path)


def test_load_tasks_rejects_session_dataset_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "t1",
                "category": "multi_turn",
                "dataset": "orders.csv",
                "question": "第一轮",
                "expected": 1,
                "session": "s1",
            },
            {
                "id": "t2",
                "category": "multi_turn",
                "dataset": "monthly_sales.csv",
                "question": "第二轮",
                "expected": 1,
                "session": "s1",
            },
        ],
    )

    with pytest.raises(ValueError, match="必须使用同一数据集"):
        load_tasks(path)


def test_repository_anomaly_expectations_are_detected() -> None:
    root = Path(__file__).resolve().parents[2]
    tasks = load_tasks(root / "eval" / "tasks.jsonl")
    expectations = load_anomaly_expectations(
        root / "eval" / "anomaly_expectations.json"
    )

    outcomes = evaluate_anomalies(tasks, root / "eval" / "datasets", expectations)

    assert len(outcomes) == 3
    for outcome in outcomes:
        assert outcome.expected_anomalies == 4
        assert outcome.detected_anomalies == outcome.expected_anomalies


def test_repository_tasks_match_independent_builder() -> None:
    from eval.build_tasks import build_tasks

    root = Path(__file__).resolve().parents[2]
    generated = build_tasks()
    loaded = load_tasks(root / "eval" / "tasks.jsonl")

    assert len(generated) == len(loaded)
    for raw, task in zip(generated, loaded, strict=True):
        assert task == EvalTask(
            id=raw["id"],
            category=raw["category"],
            dataset=raw["dataset"],
            question=raw["question"],
            expected=raw["expected"],
            expects_clarification=raw["expects_clarification"],
            session=raw.get("session"),
        )


def test_with_anomaly_metric_uses_dataset_profiles() -> None:
    root = Path(__file__).resolve().parents[2]
    tasks = load_tasks(root / "eval" / "tasks.jsonl")
    expectations = load_anomaly_expectations(
        root / "eval" / "anomaly_expectations.json"
    )
    metrics = summarize(
        [
            TaskOutcome(
                task_id="t1",
                category="aggregation",
                first_pass_ok=True,
                final_ok=True,
            )
        ]
    )

    merged = with_anomaly_metric(
        metrics, tasks, root / "eval" / "datasets", expectations
    )

    metric = merged["anomaly_detection_rate"]
    assert metric.value == 1.0
    assert metric.numerator == 12
    assert metric.denominator == 12


def test_load_outcomes_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.jsonl"
    write_jsonl(
        path,
        [
            {
                "task_id": "t1",
                "category": "aggregation",
                "first_pass_ok": True,
                "final_ok": True,
            }
        ],
    )

    outcomes = load_outcomes(path)

    assert outcomes == [
        TaskOutcome(
            task_id="t1",
            category="aggregation",
            first_pass_ok=True,
            final_ok=True,
        )
    ]


def test_load_outcomes_rejects_invalid_fields(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.jsonl"
    write_jsonl(
        path,
        [
            {
                "task_id": "t1",
                "category": "aggregation",
                "first_pass_ok": True,
                "final_ok": True,
                "unexpected": 1,
            }
        ],
    )

    with pytest.raises(ValueError, match="结果字段非法"):
        load_outcomes(path)


def test_format_metric_table_shows_pass_fail_and_na() -> None:
    metrics = summarize(
        [
            TaskOutcome(
                task_id="t1",
                category="aggregation",
                first_pass_ok=True,
                final_ok=True,
            ),
            TaskOutcome(
                task_id="t2",
                category="aggregation",
                first_pass_ok=False,
                final_ok=False,
            ),
        ]
    )

    output = format_metric_table(metrics)

    assert "一次执行成功率" in output
    assert "50.0%" in output
    assert "FAIL" in output
    assert "N/A" in output


def test_main_returns_two_when_dataset_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    write_jsonl(
        tasks_path,
        [
            {
                "id": "t1",
                "category": "aggregation",
                "dataset": "orders.csv",
                "question": "销售额是多少？",
                "expected": 100,
            }
        ],
    )
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()

    assert main(["--tasks", str(tasks_path), "--datasets-dir", str(datasets_dir)]) == 2
    assert "缺少数据集: orders.csv" in capsys.readouterr().err


def test_main_without_results_validates_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    write_jsonl(
        tasks_path,
        [
            {
                "id": "t1",
                "category": "aggregation",
                "dataset": "orders.csv",
                "question": "销售额是多少？",
                "expected": 100,
            }
        ],
    )
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    (datasets_dir / "orders.csv").write_text("order_id\n1\n", encoding="utf-8")

    assert main(["--tasks", str(tasks_path), "--datasets-dir", str(datasets_dir)]) == 0
    assert "仅完成任务与数据集校验" in capsys.readouterr().out


def test_main_with_results_outputs_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    write_jsonl(
        tasks_path,
        [
            {
                "id": "t1",
                "category": "aggregation",
                "dataset": "orders.csv",
                "question": "销售额是多少？",
                "expected": 100,
            }
        ],
    )
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    (datasets_dir / "orders.csv").write_text("order_id\n1\n", encoding="utf-8")
    results_path = tmp_path / "outcomes.jsonl"
    write_jsonl(
        results_path,
        [
            {
                "task_id": "t1",
                "category": "aggregation",
                "first_pass_ok": True,
                "final_ok": True,
            }
        ],
    )

    exit_code = main(
        [
            "--tasks",
            str(tasks_path),
            "--datasets-dir",
            str(datasets_dir),
            "--results",
            str(results_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "一次执行成功率" in output
    assert "100.0%" in output


def test_main_returns_one_when_metric_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    write_jsonl(
        tasks_path,
        [
            {
                "id": "t1",
                "category": "aggregation",
                "dataset": "orders.csv",
                "question": "销售额是多少？",
                "expected": 100,
            }
        ],
    )
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    (datasets_dir / "orders.csv").write_text("order_id\n1\n", encoding="utf-8")
    results_path = tmp_path / "outcomes.jsonl"
    write_jsonl(
        results_path,
        [
            {
                "task_id": "t1",
                "category": "aggregation",
                "first_pass_ok": False,
                "final_ok": False,
            }
        ],
    )

    exit_code = main(
        [
            "--tasks",
            str(tasks_path),
            "--datasets-dir",
            str(datasets_dir),
            "--results",
            str(results_path),
        ]
    )

    assert exit_code == 1
    assert "FAIL" in capsys.readouterr().out
