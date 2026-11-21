"""澄清门禁：体检异常未确认口径、问题模糊多解时先反问，禁止擅自决策。

门禁在生成 SQL 之前判断，命中时 Orchestrator 不进入规划与执行。
异常口径按类型在会话内确认一次；模糊问题按问题文本单独确认。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from prizm.llm.adapter import LLMAdapter, LLMError, Message
from prizm.profiling.models import Anomaly, ProfileReport

AMBIGUOUS_QUESTION = "ambiguous_question"

_POLICY_NAMES = {
    "exact_duplicate_rows": "完全重复行",
    "duplicate_primary_key": "主键重复",
    "missing_values": "缺失值",
    "dirty_numeric_format": "数值脏格式",
    AMBIGUOUS_QUESTION: "问题关注点",
}

_AGGREGATION_KEYWORDS = (
    "统计",
    "合计",
    "总计",
    "总和",
    "汇总",
    "求和",
    "计算",
    "平均",
    "均值",
    "中位",
    "计数",
    "多少",
    "数量",
    "销售额",
    "销量",
    "总额",
    "金额",
    "收入",
    "利润",
    "占比",
    "比例",
    "比率",
    "排名",
    "排行",
    "排序",
    "趋势",
    "环比",
    "同比",
    "最高",
    "最低",
    "最大",
    "最小",
    "去重",
    "清洗",
    "剔除",
    "缺失",
    "重复",
    "异常",
    "分布",
    "增长",
    "下降",
    "按渠道",
    "按产品",
    "按地区",
    "按月",
    "按天",
    "按时间",
)

_VAGUE_MARKERS = (
    "分析",
    "看看",
    "看一下",
    "看下",
    "怎么样",
    "如何",
    "有什么",
    "洞察",
    "发现",
    "总结",
    "概览",
    "概况",
    "整体",
    "情况",
)

_DUPLICATE_CHOICES = (
    ("dedupe_keep_first", "去重（保留首条）", "删除重复记录，只保留第一次出现"),
    ("keep_all", "保留全部原始行", "不做去重，按原始记录统计"),
)

_MISSING_CHOICES = (
    ("drop_missing", "剔除含缺失的行", "统计前删除涉及缺失值的行"),
    ("fill_zero", "缺失按 0 处理", "把缺失值当作 0 参与计算"),
    ("keep_missing", "保留缺失原样", "不填充，SQL 中按 NULL 处理"),
)

_DIRTY_CHOICES = (
    ("clean_numeric", "清洗为数值后统计", "去掉千分位/货币符/单位后再计算"),
    ("keep_raw", "按原始文本处理", "不做数值清洗，按原样读取"),
)

_GENERIC_CHOICES = (
    ("keep_all", "按原样处理", "不做额外处理，按原始数据统计"),
    ("clean_numeric", "先清洗再统计", "先做数据清洗再统计"),
)

_CODE_CHOICES = {
    "exact_duplicate_rows": _DUPLICATE_CHOICES,
    "duplicate_primary_key": _DUPLICATE_CHOICES,
    "missing_values": _MISSING_CHOICES,
    "dirty_numeric_format": _DIRTY_CHOICES,
}

_AMBIGUITY_CHOICES = (
    ("overview", "先看数据概览", "先展示行列数、类型与样本"),
    ("trend", "关注时间趋势", "按时间维度分析变化趋势"),
    ("dimension", "关注维度对比", "按渠道/产品/地区等维度对比"),
    ("quality", "先排查数据质量", "优先检查缺失、重复与脏格式"),
)

_LLM_AMBIGUITY_PROMPT = """你是数据分析的澄清门禁。
判断用户问题是否存在多种解读、缺少明确指标或维度。
只输出一个 JSON 对象，不要输出解释或 Markdown，结构：
{"ambiguous": true, "reason": "为什么需要澄清",
 "question": "向用户的反问", "options": ["关注方向1", "关注方向2"]}
规则：
- 问题明确（含具体指标、维度或筛选条件）时 ambiguous 必须是 false。
- 问题模糊（如“分析一下这数据”）时 ambiguous 为 true，并给出 2-4 个关注方向。
- 问题不是数据分析问题、或已足够明确时，一律 ambiguous=false。"""


@dataclass(frozen=True, slots=True)
class ClarificationOption:
    value: str
    label: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class ClarificationRequest:
    code: str
    reason: str
    question: str
    options: tuple[ClarificationOption, ...]
    column: str | None = None

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("code 不能为空")
        if not self.options:
            raise ValueError("澄清请求必须包含至少一个选项")


@dataclass(frozen=True, slots=True)
class ClarifyDecision:
    requests: tuple[ClarificationRequest, ...] = ()

    @property
    def needs_clarification(self) -> bool:
        return bool(self.requests)


class AmbiguityJudge(Protocol):
    def judge(
        self,
        question: str,
        profile: ProfileReport,
    ) -> ClarificationRequest | None: ...


class AmbiguityParseError(ValueError):
    """LLM 的模糊判定输出无法解析。"""


@dataclass
class LLMAmbiguityJudge:
    """用 LLM 判断问题是否模糊；解析失败由调用方回退到启发式。"""

    llm: LLMAdapter

    def judge(self, question: str, profile: ProfileReport) -> ClarificationRequest | None:
        return parse_ambiguity_response(
            self.llm.complete(build_ambiguity_messages(question, profile)).text,
            question=question,
        )


class ClarifyGate:
    """在规划前判定是否需要澄清；按会话记录已确认口径。"""

    def __init__(
        self,
        *,
        ambiguity_judge: AmbiguityJudge | None = None,
        confirmed: Mapping[str, str] | None = None,
    ) -> None:
        self._ambiguity_judge = ambiguity_judge
        self._confirmed: dict[str, str] = dict(confirmed or {})
        self._confirmed_questions: dict[str, str] = {}
        self._last_question = ""

    @property
    def confirmed(self) -> dict[str, str]:
        return dict(self._confirmed)

    def policy_summary(self) -> dict[str, str]:
        """已确认口径的可读摘要，供 planner prompt 注入。"""
        summary = {
            _POLICY_NAMES.get(code, code): choice for code, choice in self._confirmed.items()
        }
        if self._last_question in self._confirmed_questions:
            summary[_POLICY_NAMES[AMBIGUOUS_QUESTION]] = self._confirmed_questions[
                self._last_question
            ]
        return summary

    def judge(self, question: str, profile: ProfileReport) -> ClarifyDecision:
        text = question.strip()
        self._last_question = text
        requests: list[ClarificationRequest] = []
        for code, anomalies in _group_anomalies(profile.anomalies).items():
            if code in self._confirmed:
                continue
            if not _group_is_relevant(text, anomalies):
                continue
            requests.append(_build_anomaly_request(code, anomalies))
        if not requests and text not in self._confirmed_questions:
            ambiguity = self._ambiguity_request(text, profile)
            if ambiguity is not None:
                requests.append(ambiguity)
        return ClarifyDecision(tuple(requests))

    def confirm(self, code: str, choice: str, *, question: str | None = None) -> None:
        if not code or not choice:
            raise ValueError("确认项与口径不能为空")
        if code == AMBIGUOUS_QUESTION:
            target = question if question is not None else self._last_question
            if not target.strip():
                raise ValueError("模糊问题确认缺少问题文本")
            self._confirmed_questions[target.strip()] = choice
            return
        self._confirmed[code] = choice

    def _ambiguity_request(
        self,
        question: str,
        profile: ProfileReport,
    ) -> ClarificationRequest | None:
        if self._ambiguity_judge is not None:
            try:
                return self._ambiguity_judge.judge(question, profile)
            except (LLMError, ValueError):
                # DEBT-011: LLM 模糊判定失败时回退关键词启发式, 可能漏判或误判。
                pass
        if _looks_ambiguous(question):
            # DEBT-010: 模糊问题仅反问关注点, 未自动展示数据概览。
            return ClarificationRequest(
                code=AMBIGUOUS_QUESTION,
                reason="问题较模糊，缺少明确的指标或维度",
                question="这个问题有多种解读，你想重点关注哪些方面？",
                options=tuple(
                    ClarificationOption(value, label, description)
                    for value, label, description in _AMBIGUITY_CHOICES
                ),
            )
        return None


def build_ambiguity_messages(question: str, profile: ProfileReport) -> list[Message]:
    columns = "、".join(column.name for column in profile.columns[:30]) or "（无列信息）"
    return [
        {"role": "system", "content": _LLM_AMBIGUITY_PROMPT},
        {
            "role": "user",
            "content": f"数据列：{columns}\n用户问题：{question.strip()}",
        },
    ]


def parse_ambiguity_response(
    text: str,
    *,
    question: str,
) -> ClarificationRequest | None:
    raw = text.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end < start:
        raise AmbiguityParseError("澄清判定未返回 JSON 对象")
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise AmbiguityParseError(f"澄清判定 JSON 解析失败: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise AmbiguityParseError("澄清判定必须是 JSON 对象")

    ambiguous = payload.get("ambiguous")
    if not isinstance(ambiguous, bool):
        raise AmbiguityParseError("ambiguous 必须是布尔值")
    if not ambiguous:
        return None

    reason = payload.get("reason")
    ask = payload.get("question")
    options = _parse_options(payload.get("options"))
    return ClarificationRequest(
        code=AMBIGUOUS_QUESTION,
        reason=reason.strip() if isinstance(reason, str) and reason.strip() else "问题存在多种解读",
        question=(
            ask.strip()
            if isinstance(ask, str) and ask.strip()
            else "这个问题有多种解读，你想重点关注哪些方面？"
        ),
        options=options,
    )


def _parse_options(raw: object) -> tuple[ClarificationOption, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return _default_ambiguity_options()
    labels = [item.strip() for item in raw if isinstance(item, str) and item.strip()]
    if not labels:
        return _default_ambiguity_options()
    return tuple(
        ClarificationOption(value=f"option_{index}", label=label)
        for index, label in enumerate(labels[:4], start=1)
    )


def _default_ambiguity_options() -> tuple[ClarificationOption, ...]:
    return tuple(
        ClarificationOption(value, label, description)
        for value, label, description in _AMBIGUITY_CHOICES
    )


def _looks_ambiguous(question: str) -> bool:
    text = question.strip()
    if not text:
        return False
    if any(keyword in text for keyword in _AGGREGATION_KEYWORDS):
        return False
    return any(marker in text for marker in _VAGUE_MARKERS)


def _group_anomalies(anomalies: Sequence[Anomaly]) -> dict[str, tuple[Anomaly, ...]]:
    groups: dict[str, list[Anomaly]] = {}
    for anomaly in anomalies:
        groups.setdefault(anomaly.code, []).append(anomaly)
    return {code: tuple(items) for code, items in groups.items()}


def _group_is_relevant(question: str, anomalies: Sequence[Anomaly]) -> bool:
    # DEBT-009: 异常相关性按关键词与列名出现判断, 未结合 SQL 语义。
    if _looks_ambiguous(question):
        return True
    if any(keyword in question for keyword in _AGGREGATION_KEYWORDS):
        return True
    for anomaly in anomalies:
        for column in _split_columns(anomaly.column):
            if column and column in question:
                return True
    return False


def _build_anomaly_request(
    code: str,
    anomalies: Sequence[Anomaly],
) -> ClarificationRequest:
    columns = _unique_columns(anomalies)
    column_text = "、".join(columns) if columns else "相关列"
    count = sum(anomaly.count for anomaly in anomalies)
    choices = _CODE_CHOICES.get(code, _GENERIC_CHOICES)
    options = tuple(
        ClarificationOption(value, label, description) for value, label, description in choices
    )
    if code == "exact_duplicate_rows":
        reason = f"存在 {count} 行完全重复"
        question = f"检测到 {count} 行完全重复，统计前是否去重？"
    elif code == "duplicate_primary_key":
        reason = f"疑似主键 {column_text} 有 {count} 行重复"
        question = f"疑似主键 {column_text} 存在重复（共 {count} 行），按哪种口径处理？"
    elif code == "missing_values":
        reason = f"列 {column_text} 共 {count} 个缺失值"
        question = f"列 {column_text} 存在缺失值（共 {count} 个），统计前如何处理？"
    elif code == "dirty_numeric_format":
        reason = f"列 {column_text} 有 {count} 个值混入千分位/单位"
        question = f"列 {column_text} 存在混入千分位/单位的值（共 {count} 个），是否先清洗为数值？"
    else:
        reason = f"检测到 {code} 异常"
        question = f"检测到数据异常（{code}，共 {count} 处），按哪种口径处理？"
    return ClarificationRequest(
        code=code,
        reason=reason,
        question=question,
        options=options,
        column=column_text,
    )


def _unique_columns(anomalies: Sequence[Anomaly]) -> list[str]:
    columns: list[str] = []
    for anomaly in anomalies:
        for column in _split_columns(anomaly.column):
            if column and column not in columns:
                columns.append(column)
    return columns


def _split_columns(column: str | None) -> list[str]:
    if not column:
        return []
    return [part for part in column.split("/") if part]
