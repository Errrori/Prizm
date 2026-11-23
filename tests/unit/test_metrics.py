from typing import Any

import pytest
from eval.metrics import (
    TARGET_ANOMALY_DETECTION,
    TARGET_CLARIFICATION_ACCURACY,
    TARGET_FINAL_SUCCESS,
    TARGET_FIRST_PASS_SUCCESS,
    TARGET_RAG_CITATION_ACCURACY,
    EvalTask,
    TaskOutcome,
    anomaly_detection_rate,
    clarification_accuracy,
    final_success_rate,
    first_pass_success_rate,
    parse_task,
    rag_citation_accuracy,
    summarize,
)


def make_outcome(**overrides: Any) -> TaskOutcome:
    values: dict[str, Any] = {
        "task_id": "task-1",
        "category": "aggregation",
        "first_pass_ok": True,
        "final_ok": True,
    }
    values.update(overrides)
    return TaskOutcome(**values)


def test_parse_task_returns_task_with_defaults() -> None:
    task = parse_task(
        {
            "id": "orders-001",
            "category": "aggregation",
            "dataset": "orders.csv",
            "question": "各渠道销售额是多少？",
            "expected": {"channel": "A", "sales": 100},
        }
    )

    assert task == EvalTask(
        id="orders-001",
        category="aggregation",
        dataset="orders.csv",
        question="各渠道销售额是多少？",
        expected={"channel": "A", "sales": 100},
        expects_clarification=False,
    )


def test_parse_task_keeps_explicit_clarification_flag() -> None:
    task = parse_task(
        {
            "id": "orders-002",
            "category": "cleaning",
            "dataset": "orders_dirty.csv",
            "question": "按订单去重后统计销售额",
            "expected": 100,
            "expects_clarification": True,
        }
    )

    assert task.expects_clarification is True


def test_parse_task_reads_session() -> None:
    task = parse_task(
        {
            "id": "orders-006",
            "category": "multi_turn",
            "dataset": "orders.csv",
            "question": "只看已完成订单，再按渠道统计销售额。",
            "expected": [],
            "session": "orders-followup",
        }
    )

    assert task.session == "orders-followup"


def test_parse_task_rejects_blank_session() -> None:
    with pytest.raises(ValueError, match="session 必须是非空字符串"):
        parse_task(
            {
                "id": "orders-007",
                "category": "multi_turn",
                "dataset": "orders.csv",
                "question": "再统计一次",
                "expected": [],
                "session": "  ",
            }
        )


def test_parse_task_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="任务缺少字段: category, question"):
        parse_task(
            {
                "id": "orders-003",
                "dataset": "orders.csv",
                "expected": 1,
            }
        )


def test_parse_task_rejects_unknown_category() -> None:
    with pytest.raises(ValueError, match="未知 category"):
        parse_task(
            {
                "id": "orders-004",
                "category": "unknown",
                "dataset": "orders.csv",
                "question": "销售额是多少？",
                "expected": 1,
            }
        )


def test_parse_task_rejects_non_bool_clarification_flag() -> None:
    with pytest.raises(ValueError, match="expects_clarification 必须是 bool"):
        parse_task(
            {
                "id": "orders-005",
                "category": "aggregation",
                "dataset": "orders.csv",
                "question": "销售额是多少？",
                "expected": 1,
                "expects_clarification": "yes",
            }
        )


def test_first_pass_success_rate() -> None:
    outcomes = [
        make_outcome(task_id="1", first_pass_ok=True),
        make_outcome(task_id="2", first_pass_ok=False),
        make_outcome(task_id="3", first_pass_ok=True),
        make_outcome(task_id="4", first_pass_ok=False),
    ]

    metric = first_pass_success_rate(outcomes)

    assert metric.value == 0.5
    assert metric.numerator == 2
    assert metric.denominator == 4
    assert metric.target == TARGET_FIRST_PASS_SUCCESS
    assert metric.passed is False


def test_final_success_rate() -> None:
    outcomes = [
        make_outcome(task_id="1", final_ok=True),
        make_outcome(task_id="2", final_ok=True),
        make_outcome(task_id="3", final_ok=True),
        make_outcome(task_id="4", final_ok=False),
    ]

    metric = final_success_rate(outcomes)

    assert metric.value == 0.75
    assert metric.passed is False
    assert metric.target == TARGET_FINAL_SUCCESS


def test_anomaly_detection_rate() -> None:
    outcomes = [
        make_outcome(expected_anomalies=2, detected_anomalies=2),
        make_outcome(expected_anomalies=1, detected_anomalies=1),
    ]

    metric = anomaly_detection_rate(outcomes)

    assert metric.value == 1.0
    assert metric.passed is True
    assert metric.target == TARGET_ANOMALY_DETECTION


def test_anomaly_detection_rate_reports_missed_anomalies() -> None:
    outcomes = [
        make_outcome(expected_anomalies=2, detected_anomalies=1),
        make_outcome(expected_anomalies=1, detected_anomalies=0),
    ]

    metric = anomaly_detection_rate(outcomes)

    assert metric.numerator == 1
    assert metric.denominator == 3
    assert metric.value == pytest.approx(1 / 3)
    assert metric.passed is False


def test_clarification_accuracy_counts_ask_and_do_not_ask() -> None:
    outcomes = [
        make_outcome(expected_clarification=True, clarification_triggered=True),
        make_outcome(expected_clarification=False, clarification_triggered=False),
        make_outcome(expected_clarification=True, clarification_triggered=False),
        make_outcome(expected_clarification=False, clarification_triggered=True),
    ]

    metric = clarification_accuracy(outcomes)

    assert metric.value == 0.5
    assert metric.target == TARGET_CLARIFICATION_ACCURACY
    assert metric.passed is False


def test_rag_citation_accuracy() -> None:
    outcomes = [
        make_outcome(rag_citations_total=2, rag_citations_correct=2),
        make_outcome(rag_citations_total=8, rag_citations_correct=7),
    ]

    metric = rag_citation_accuracy(outcomes)

    assert metric.numerator == 9
    assert metric.denominator == 10
    assert metric.value == pytest.approx(9 / 10)
    assert metric.target == TARGET_RAG_CITATION_ACCURACY
    assert metric.passed is True


def test_empty_outcomes_have_no_value_and_are_not_passed() -> None:
    metrics = summarize([])

    assert set(metrics) == {
        "first_pass_success_rate",
        "final_success_rate",
        "anomaly_detection_rate",
        "clarification_accuracy",
        "rag_citation_accuracy",
    }
    assert all(metric.value is None for metric in metrics.values())
    assert all(metric.passed is None for metric in metrics.values())


def test_summarize_wires_requirement_targets() -> None:
    metrics = summarize([make_outcome()])

    assert metrics["first_pass_success_rate"].target == TARGET_FIRST_PASS_SUCCESS
    assert metrics["final_success_rate"].target == TARGET_FINAL_SUCCESS
    assert metrics["anomaly_detection_rate"].target == TARGET_ANOMALY_DETECTION
    assert metrics["clarification_accuracy"].target == TARGET_CLARIFICATION_ACCURACY
    assert metrics["rag_citation_accuracy"].target == TARGET_RAG_CITATION_ACCURACY


def test_task_outcome_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="detected_anomalies"):
        make_outcome(expected_anomalies=1, detected_anomalies=2)

    with pytest.raises(ValueError, match="rag_citations_correct"):
        make_outcome(rag_citations_total=1, rag_citations_correct=2)
