import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest
from eval.metrics import EvalTask, TaskOutcome
from eval.run_eval import (
    execute_tasks,
    load_outcomes,
    main,
    matches_expected,
    write_outcomes,
)

from prizm.config import Settings
from prizm.llm.adapter import Completion, Message


@dataclass
class ScriptedLLM:
    responses: list[str]
    calls: list[list[Message]] = field(default_factory=list)

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> Completion:
        self.calls.append([dict(message) for message in messages])
        if not self.responses:
            raise AssertionError("没有更多脚本化响应")
        return Completion(self.responses.pop(0))


def plan(sql: str) -> str:
    return json.dumps({"sql": sql, "chart": None})


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_orders(path: Path) -> None:
    pd.DataFrame(
        {
            "status": ["已完成", "已取消", "已完成"],
            "sales": [100, 999, 200],
        }
    ).to_csv(path, index=False, encoding="utf-8")


def make_task(
    task_id: str,
    *,
    question: str = "已完成订单的销售额是多少？",
    expected: object = 300,
    dataset: str = "orders.csv",
    expects_clarification: bool = False,
) -> EvalTask:
    return EvalTask(
        id=task_id,
        category="aggregation",
        dataset=dataset,
        question=question,
        expected=expected,
        expects_clarification=expects_clarification,
    )


def test_matches_expected_scalar() -> None:
    assert matches_expected(pd.DataFrame({"total": [300]}), 300) is True
    assert matches_expected(pd.DataFrame({"total": [300.0]}), 300) is True
    assert matches_expected(pd.DataFrame({"total": ["300"]}), 300) is True
    assert matches_expected(pd.DataFrame({"total": [300, 400]}), 300) is False
    assert matches_expected(pd.DataFrame({"total": [300], "extra": [1]}), 300) is False
    assert matches_expected(pd.DataFrame({"total": [300]}), None) is False


def test_matches_expected_records_allows_extra_columns() -> None:
    frame = pd.DataFrame(
        {
            "channel": ["分销", "线上"],
            "sales": [4600, 3800],
            "extra": [1, 2],
        }
    )
    expected = [
        {"channel": "分销", "sales": 4600},
        {"channel": "线上", "sales": 3800},
    ]

    assert matches_expected(frame, expected) is True

    reversed_expected = list(reversed(expected))
    assert matches_expected(frame, reversed_expected) is False


def test_matches_expected_records_requires_expected_columns() -> None:
    frame = pd.DataFrame({"channel": ["分销"], "sales": [4600]})

    assert matches_expected(frame, [{"channel": "分销"}]) is True
    assert matches_expected(frame, [{"channel": "分销", "region": "华北"}]) is False


def test_matches_expected_handles_date_and_missing_values() -> None:
    frame = pd.DataFrame({"month": [pd.Timestamp("2026-03-01")]})

    assert matches_expected(frame, "2026-03") is True
    assert matches_expected(pd.DataFrame({"value": [None]}), None) is False


def test_execute_tasks_tracks_first_pass_and_repair(tmp_path: Path) -> None:
    write_orders(tmp_path / "orders.csv")
    tasks = [
        make_task("first-pass"),
        make_task("after-repair"),
    ]
    good_sql = "SELECT SUM(sales) AS total FROM orders WHERE status = '已完成'"
    llm = ScriptedLLM(
        [
            plan(good_sql),
            plan("SELECT missing_column FROM orders"),
            plan(good_sql),
        ]
    )

    outcomes = execute_tasks(tasks, tmp_path, llm, max_attempts=3)

    assert outcomes[0].first_pass_ok is True
    assert outcomes[0].final_ok is True
    assert outcomes[0].attempts == 1
    assert outcomes[0].error == ""

    assert outcomes[1].first_pass_ok is False
    assert outcomes[1].final_ok is True
    assert outcomes[1].attempts == 2
    assert outcomes[1].error == ""
    assert len(llm.calls) == 3


def test_execute_tasks_records_clarification_outcomes() -> None:
    root = Path(__file__).resolve().parents[2]
    datasets_dir = root / "eval" / "datasets"
    question = (
        "检测到完全重复行，也有同一订单号对应不同金额的记录。"
        "统计销售额前是否应该去重？按哪种去重口径？"
    )
    tasks = [
        make_task(
            "expected-clarification",
            question=question,
            expected=None,
            dataset="orders_dirty.csv",
            expects_clarification=True,
        ),
        make_task(
            "unexpected-clarification",
            question=question,
            expected=None,
            dataset="orders_dirty.csv",
            expects_clarification=False,
        ),
    ]
    llm = ScriptedLLM([])

    outcomes = execute_tasks(tasks, datasets_dir, llm)

    assert outcomes[0].clarification_triggered is True
    assert outcomes[0].first_pass_ok is True
    assert outcomes[0].final_ok is True
    assert outcomes[0].error == ""

    assert outcomes[1].clarification_triggered is True
    assert outcomes[1].first_pass_ok is False
    assert outcomes[1].final_ok is False
    assert outcomes[1].error == "未预期的澄清"
    assert llm.calls == []


def test_execute_tasks_records_exhausted_repair(tmp_path: Path) -> None:
    write_orders(tmp_path / "orders.csv")
    tasks = [make_task("fails")]
    llm = ScriptedLLM([plan("SELECT missing_column FROM orders")] * 2)

    outcomes = execute_tasks(tasks, tmp_path, llm, max_attempts=2)

    assert outcomes[0].first_pass_ok is False
    assert outcomes[0].final_ok is False
    assert outcomes[0].attempts == 2
    assert "missing_column" in outcomes[0].error
    assert len(llm.calls) == 2


def test_execute_tasks_reuses_session_context(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "status": ["已完成", "已完成", "已取消"],
            "channel": ["线上", "线下", "线上"],
            "sales": [100, 200, 999],
        }
    ).to_csv(tmp_path / "orders.csv", index=False, encoding="utf-8")
    tasks = [
        EvalTask(
            id="turn-1",
            category="multi_turn",
            dataset="orders.csv",
            question="已完成订单的销售额是多少？",
            expected=300,
            session="orders-session",
        ),
        EvalTask(
            id="turn-2",
            category="multi_turn",
            dataset="orders.csv",
            question="只看线上渠道，再算一次销售额。",
            expected=100,
            session="orders-session",
        ),
    ]
    first_sql = "SELECT SUM(sales) AS total FROM orders WHERE status = '已完成'"
    second_sql = (
        "SELECT SUM(sales) AS total FROM orders "
        "WHERE status = '已完成' AND channel = '线上'"
    )
    llm = ScriptedLLM([plan(first_sql), plan(second_sql)])

    outcomes = execute_tasks(tasks, tmp_path, llm)

    assert all(outcome.final_ok for outcome in outcomes)
    assert len(llm.calls) == 2
    second_prompt = llm.calls[1][-1]["content"]
    assert "最近对话与结果" in second_prompt
    assert first_sql in second_prompt
    assert "第 1 轮问题" in second_prompt


def test_write_outcomes_round_trips(tmp_path: Path) -> None:
    outcomes = [
        TaskOutcome(
            task_id="t1",
            category="aggregation",
            first_pass_ok=False,
            final_ok=True,
            attempts=2,
        ),
        TaskOutcome(
            task_id="t2",
            category="cleaning",
            first_pass_ok=True,
            final_ok=True,
            expected_clarification=True,
            clarification_triggered=True,
        ),
    ]
    path = tmp_path / "nested" / "outcomes.jsonl"

    write_outcomes(path, outcomes)

    assert load_outcomes(path) == outcomes
    assert "attempts" in path.read_text(encoding="utf-8")


def test_main_rejects_execute_with_results() -> None:
    assert main(["--execute", "--results", "outcomes.jsonl"]) == 2


def test_main_rejects_limit_without_execute() -> None:
    assert main(["--limit", "1"]) == 2


def test_main_rejects_invalid_max_attempts() -> None:
    assert main(["--execute", "--max-attempts", "4"]) == 2


def test_main_execute_rejects_mock_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("eval.run_eval.get_settings", lambda: Settings(default_model="mock"))

    assert main(["--execute", "--limit", "1"]) == 2
    assert "mock" in capsys.readouterr().err


def test_main_execute_writes_outcomes_and_prints_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    write_jsonl(
        tasks_path,
        [
            {
                "id": "t1",
                "category": "aggregation",
                "dataset": "orders.csv",
                "question": "已完成订单的销售额是多少？",
                "expected": 300,
            }
        ],
    )
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    write_orders(datasets_dir / "orders.csv")
    outcomes_path = tmp_path / "outcomes.jsonl"

    settings = Settings(default_model="openai/gpt-4o-mini")
    llm = ScriptedLLM(
        [plan("SELECT SUM(sales) AS total FROM orders WHERE status = '已完成'")]
    )
    monkeypatch.setattr("eval.run_eval.get_settings", lambda: settings)
    monkeypatch.setattr("eval.run_eval.build_llm", lambda *args, **kwargs: llm)

    exit_code = main(
        [
            "--tasks",
            str(tasks_path),
            "--datasets-dir",
            str(datasets_dir),
            "--execute",
            "--limit",
            "1",
            "--outcomes",
            str(outcomes_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "结果已写入" in output
    assert "一次执行成功率" in output
    assert load_outcomes(outcomes_path)[0].final_ok is True
