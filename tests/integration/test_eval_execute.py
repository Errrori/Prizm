import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from eval.metrics import summarize
from eval.run_eval import DEFAULT_DATASETS_DIR, DEFAULT_TASKS_PATH, execute_tasks, load_tasks

from prizm.llm.adapter import Completion, Message

REPO_ROOT = Path(__file__).resolve().parents[2]

SQL_BY_QUESTION = {
    "这份订单数据里是否存在完全重复的行？如果存在，返回重复行数。": (
        "SELECT 0 AS duplicate_rows"
    ),
    "统计 2026 年第一季度状态为“已完成”的订单销售额总额。": (
        "SELECT SUM(sales) AS total FROM orders "
        "WHERE status = '已完成' "
        "AND order_date >= '2026-01-01' AND order_date < '2026-04-01'"
    ),
    "按渠道统计状态为“已完成”的订单销售额，并按销售额从高到低排序。": (
        "SELECT channel, SUM(sales) AS sales FROM orders "
        "WHERE status = '已完成' GROUP BY channel ORDER BY sales DESC"
    ),
    "按月份统计状态为“已完成”的订单销售额趋势。": (
        "SELECT substring(order_date, 1, 7) AS month, SUM(sales) AS sales FROM orders "
        "WHERE status = '已完成' GROUP BY month ORDER BY month"
    ),
    "哪个月的已完成订单销售额最高？只返回月份。": (
        "SELECT substring(order_date, 1, 7) AS month FROM orders "
        "WHERE status = '已完成' GROUP BY month ORDER BY SUM(sales) DESC LIMIT 1"
    ),
    "哪个渠道对已完成订单销售额贡献最大？只返回渠道名称。": (
        "SELECT channel FROM orders WHERE status = '已完成' "
        "GROUP BY channel ORDER BY SUM(sales) DESC LIMIT 1"
    ),
    "承接上一题的筛选条件，只看已完成订单，按产品统计销售额，并从高到低排序。": (
        "SELECT product, SUM(sales) AS sales FROM orders "
        "WHERE status = '已完成' GROUP BY product ORDER BY sales DESC"
    ),
}


class OracleLLM:
    """用标准 SQL 模拟“理想模型”，验证评测执行器与答案比对。"""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> Completion:
        self.calls += 1
        prompt = messages[-1]["content"]
        question = prompt.rsplit("问题：", 1)[-1].strip()
        try:
            sql = SQL_BY_QUESTION[question]
        except KeyError as exc:
            raise AssertionError(f"没有为问题准备标准 SQL: {question!r}") from exc
        return Completion(json.dumps({"sql": sql, "chart": None}, ensure_ascii=False))


@pytest.mark.integration
def test_minimal_tasks_can_reach_full_success_with_oracle_llm() -> None:
    tasks = load_tasks(REPO_ROOT / DEFAULT_TASKS_PATH)
    llm = OracleLLM()

    outcomes = execute_tasks(tasks, REPO_ROOT / DEFAULT_DATASETS_DIR, llm)

    assert [outcome.task_id for outcome in outcomes] == [task.id for task in tasks]
    assert all(outcome.first_pass_ok for outcome in outcomes)
    assert all(outcome.final_ok for outcome in outcomes)
    assert all(outcome.error == "" for outcome in outcomes)
    assert llm.calls == 7

    metrics = summarize(outcomes)
    assert metrics["first_pass_success_rate"].value == 1.0
    assert metrics["final_success_rate"].value == 1.0
    assert metrics["clarification_accuracy"].value == 1.0
