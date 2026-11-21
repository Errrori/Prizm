from eval.metrics import EvalTask
from eval.run_clarify import evaluate, format_report, main
from eval.run_eval import DEFAULT_DATASETS_DIR, DEFAULT_TASKS_PATH, load_tasks


def make_task(task_id: str, *, expects_clarification: bool) -> EvalTask:
    return EvalTask(
        id=task_id,
        category="cleaning",
        dataset="orders.csv",
        question="统计销售额",
        expected=None,
        expects_clarification=expects_clarification,
    )


def test_actual_tasks_meet_clarification_targets() -> None:
    tasks = load_tasks(DEFAULT_TASKS_PATH)

    report, passed = format_report(evaluate(tasks, DEFAULT_DATASETS_DIR))

    assert passed is True
    assert "3/3 = 100.0%" in report
    assert "10/10 = 100.0%" in report


def test_mismatch_is_reported_and_fails() -> None:
    results = [
        (make_task("missed", expects_clarification=True), False),
        (make_task("false-positive", expects_clarification=False), True),
    ]

    report, passed = format_report(results)

    assert passed is False
    assert "不一致" in report
    assert "missed" in report
    assert "false-positive" in report


def test_main_returns_zero_for_current_tasks(capsys) -> None:
    assert main([]) == 0
    assert "澄清门禁评测" in capsys.readouterr().out
