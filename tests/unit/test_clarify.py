import json

import pandas as pd
import pytest

from prizm.agent.clarify import (
    AMBIGUOUS_QUESTION,
    ClarificationOption,
    ClarificationRequest,
    ClarifyGate,
    LLMAmbiguityJudge,
    parse_ambiguity_response,
)
from prizm.llm.adapter import LLMError, MockLLM
from prizm.profiling.models import ProfileReport
from prizm.profiling.profiler import profile_frame

ALL_ANOMALY_CODES = {
    "missing_values",
    "exact_duplicate_rows",
    "duplicate_primary_key",
    "dirty_numeric_format",
}


def clean_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3],
            "channel": ["线上", "线下", "线上"],
            "sales": [100, 200, 300],
        }
    )


def dirty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [1, 1, 2, 2, 3],
            "channel": ["线上", "线上", "线下", "线下", "线上"],
            "quantity": [1, None, 2, 2, 4],
            "sales": ["1,200", "800", "800", "800", "300"],
        }
    )


class StaticJudge:
    def __init__(self, result: ClarificationRequest | Exception | None) -> None:
        self.result = result
        self.calls: list[str] = []

    def judge(
        self,
        question: str,
        profile: ProfileReport,
    ) -> ClarificationRequest | None:
        self.calls.append(question)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_clean_profile_and_concrete_question_passes() -> None:
    gate = ClarifyGate()

    decision = gate.judge("按渠道统计销售额", profile_frame(clean_frame()))

    assert decision.needs_clarification is False


def test_relevant_anomalies_produce_one_request_per_code() -> None:
    gate = ClarifyGate()

    decision = gate.judge("统计销售额", profile_frame(dirty_frame()))

    assert {request.code for request in decision.requests} == ALL_ANOMALY_CODES
    assert all(request.options for request in decision.requests)


def test_irrelevant_question_does_not_trigger_anomaly_requests() -> None:
    gate = ClarifyGate()

    decision = gate.judge("有哪些渠道", profile_frame(dirty_frame()))

    assert decision.needs_clarification is False


def test_confirmed_anomalies_are_not_asked_again() -> None:
    gate = ClarifyGate()
    profile = profile_frame(dirty_frame())
    first = gate.judge("统计销售额", profile)
    for request in first.requests:
        gate.confirm(request.code, request.options[0].label)

    second = gate.judge("统计销售额", profile)

    assert first.needs_clarification is True
    assert second.needs_clarification is False


def test_missing_values_of_multiple_columns_are_grouped() -> None:
    frame = pd.DataFrame(
        {
            "order_id": [1, 2, 3],
            "quantity": [1, None, 3],
            "status": ["已完成", "", "已完成"],
        }
    )
    gate = ClarifyGate()

    decision = gate.judge("统计缺失情况", profile_frame(frame))

    missing = [request for request in decision.requests if request.code == "missing_values"]
    assert len(missing) == 1
    assert missing[0].column is not None
    assert "quantity" in missing[0].column
    assert "status" in missing[0].column


def test_vague_question_triggers_ambiguity_request() -> None:
    gate = ClarifyGate()

    decision = gate.judge("分析一下这数据", profile_frame(clean_frame()))

    assert [request.code for request in decision.requests] == [AMBIGUOUS_QUESTION]
    assert decision.requests[0].options


def test_vague_question_on_dirty_data_checks_anomalies_first() -> None:
    gate = ClarifyGate()

    decision = gate.judge("分析一下这数据", profile_frame(dirty_frame()))

    assert {request.code for request in decision.requests} == ALL_ANOMALY_CODES


def test_concrete_question_is_not_ambiguous() -> None:
    gate = ClarifyGate()

    decision = gate.judge("按渠道统计销售额", profile_frame(clean_frame()))

    assert decision.needs_clarification is False


def test_llm_judge_result_is_used() -> None:
    request = ClarificationRequest(
        code=AMBIGUOUS_QUESTION,
        reason="模型判定模糊",
        question="你想看哪个指标？",
        options=(ClarificationOption("option_1", "销售额"),),
    )
    judge = StaticJudge(request)
    gate = ClarifyGate(ambiguity_judge=judge)

    decision = gate.judge("随便看看", profile_frame(clean_frame()))

    assert decision.requests[0].question == "你想看哪个指标？"
    assert judge.calls == ["随便看看"]


def test_llm_judge_error_falls_back_to_heuristic() -> None:
    judge = StaticJudge(LLMError("boom"))
    gate = ClarifyGate(ambiguity_judge=judge)

    decision = gate.judge("分析一下这数据", profile_frame(clean_frame()))

    assert [request.code for request in decision.requests] == [AMBIGUOUS_QUESTION]


def test_llm_judge_error_does_not_force_clarification() -> None:
    judge = StaticJudge(LLMError("boom"))
    gate = ClarifyGate(ambiguity_judge=judge)

    decision = gate.judge("按渠道统计销售额", profile_frame(clean_frame()))

    assert decision.needs_clarification is False


def test_confirmed_ambiguity_is_scoped_to_the_same_question() -> None:
    gate = ClarifyGate()
    profile = profile_frame(clean_frame())
    gate.judge("分析一下这数据", profile)
    gate.confirm(AMBIGUOUS_QUESTION, "关注时间趋势", question="分析一下这数据")

    same = gate.judge("分析一下这数据", profile)
    other = gate.judge("看看这份数据怎么样", profile)

    assert same.needs_clarification is False
    assert other.needs_clarification is True


def test_policy_summary_contains_confirmed_choices() -> None:
    gate = ClarifyGate()
    gate.confirm("missing_values", "剔除含缺失的行")
    profile = profile_frame(clean_frame())
    gate.judge("分析一下这数据", profile)
    gate.confirm(AMBIGUOUS_QUESTION, "关注时间趋势", question="分析一下这数据")

    summary = gate.policy_summary()

    assert summary["缺失值"] == "剔除含缺失的行"
    assert summary["问题关注点"] == "关注时间趋势"


def test_parse_ambiguity_false_returns_none() -> None:
    result = parse_ambiguity_response(
        json.dumps({"ambiguous": False}),
        question="按渠道统计销售额",
    )

    assert result is None


def test_parse_ambiguity_true_builds_request() -> None:
    result = parse_ambiguity_response(
        json.dumps(
            {
                "ambiguous": True,
                "reason": "缺少明确指标",
                "question": "你想看哪个指标？",
                "options": ["销售额", "销量"],
            },
            ensure_ascii=False,
        ),
        question="分析一下这数据",
    )

    assert result is not None
    assert result.code == AMBIGUOUS_QUESTION
    assert result.question == "你想看哪个指标？"
    assert [option.label for option in result.options] == ["销售额", "销量"]


def test_parse_ambiguity_invalid_json_raises() -> None:
    with pytest.raises(ValueError, match="JSON"):
        parse_ambiguity_response("not json", question="分析一下这数据")


def test_llm_ambiguity_judge_uses_model_response() -> None:
    llm = MockLLM(
        json.dumps(
            {
                "ambiguous": True,
                "reason": "缺少明确指标",
                "question": "你想看哪个指标？",
                "options": ["销售额", "销量"],
            },
            ensure_ascii=False,
        )
    )
    judge = LLMAmbiguityJudge(llm)

    result = judge.judge("分析一下这数据", profile_frame(clean_frame()))

    assert result is not None
    assert result.question == "你想看哪个指标？"
    assert llm.calls
    assert "order_id" in llm.calls[0][1]["content"]
