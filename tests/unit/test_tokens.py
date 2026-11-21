from datetime import date, timedelta

import pytest

from prizm.utils.tokens import (
    Completion,
    DailyUsageLedger,
    QuotaExceededError,
    SessionTurnCounter,
    Usage,
)


class FakeClock:
    def __init__(self, current: date) -> None:
        self.current = current

    def __call__(self) -> date:
        return self.current

    def advance(self, days: int) -> None:
        self.current += timedelta(days=days)


def test_usage_normalizes_total_and_accumulates_cost() -> None:
    usage = Usage(prompt_tokens=3, completion_tokens=4)

    assert usage.total_tokens == 7
    assert Usage.combine(Usage(cost_usd=0.5), Usage(cost_usd=0.25)).cost_usd == 0.75
    assert Usage.combine(Usage(), Usage()).cost_usd is None


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "total_tokens"])
def test_usage_rejects_negative_tokens(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        Usage(**{field: -1})


def test_usage_rejects_negative_cost() -> None:
    with pytest.raises(ValueError, match="cost_usd"):
        Usage(cost_usd=-0.1)


def test_completion_defaults_to_empty_usage() -> None:
    completion = Completion("ok")

    assert completion.text == "ok"
    assert completion.usage.total_tokens == 0


def test_ledger_records_and_reports_remaining() -> None:
    ledger = DailyUsageLedger()
    ledger.record(Usage(prompt_tokens=10, completion_tokens=5))

    assert ledger.snapshot().total_tokens == 15
    assert ledger.remaining(100) == 85
    ledger.ensure_available(100)


def test_ledger_blocks_when_budget_exhausted() -> None:
    ledger = DailyUsageLedger()
    ledger.record(Usage(total_tokens=100))

    with pytest.raises(QuotaExceededError, match="额度已用完"):
        ledger.ensure_available(100)


def test_ledger_resets_on_new_day() -> None:
    clock = FakeClock(date(2026, 1, 1))
    ledger = DailyUsageLedger(today=clock)
    ledger.record(Usage(total_tokens=50))

    clock.advance(1)

    assert ledger.snapshot().total_tokens == 0
    assert ledger.day == date(2026, 1, 2)


@pytest.mark.parametrize("budget", [-1, 1.5, True])
def test_ledger_rejects_invalid_budget(budget: object) -> None:
    ledger = DailyUsageLedger()

    with pytest.raises(ValueError, match="budget"):
        ledger.remaining(budget)  # type: ignore[arg-type]


def test_session_turn_counter_stops_at_limit() -> None:
    counter = SessionTurnCounter(2)

    assert counter.try_consume() is True
    assert counter.try_consume() is True
    assert counter.try_consume() is False
    assert counter.used == 2
    assert counter.limit == 2


def test_session_turn_counter_zero_means_unlimited() -> None:
    counter = SessionTurnCounter(0)

    assert all(counter.try_consume() for _ in range(50))


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_session_turn_counter_rejects_invalid_limit(limit: object) -> None:
    with pytest.raises(ValueError, match="limit"):
        SessionTurnCounter(limit)  # type: ignore[arg-type]
