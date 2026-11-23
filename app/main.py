"""Chainlit 入口：上传 CSV → 注册 DuckDB → 体检 → 澄清门禁 → 提问 → 展示结果。"""

from __future__ import annotations

from typing import Any

import chainlit as cl
import duckdb

from prizm.agent.clarify import ClarifyGate, LLMAmbiguityJudge
from prizm.agent.orchestrator import (
    Answer,
    Clarification,
    Orchestrator,
    format_result_table,
)
from prizm.agent.planner import Planner, build_mock_plan_response
from prizm.agent.repair import RepairFailed
from prizm.agent.trace import Trace
from prizm.config import get_settings
from prizm.datasource.csv_connector import CsvConnector
from prizm.engine.duckdb_engine import DuckDBEngine, QueryTimeoutError
from prizm.llm.adapter import LLMError, build_llm
from prizm.profiling.profiler import format_profile_report
from prizm.rag.ingest import DictionaryEntry, build_data_dictionary, load_dictionary_file
from prizm.rag.retriever import DataDictionaryRetriever, DictionaryAnswer, DictionaryQA
from prizm.utils.tokens import DailyUsageLedger, QuotaExceededError, SessionTurnCounter

_usage_ledger = DailyUsageLedger()


@cl.on_chat_start
async def on_chat_start() -> None:
    settings = get_settings()
    files = await cl.AskFileMessage(
        content="请上传一个 CSV 文件，可选同时上传 .md/.txt 数据说明文档。",
        accept=["text/csv", "text/markdown", "text/plain"],
        max_files=2,
        max_size_mb=50,
    ).send()
    if not files:
        await cl.Message(content="未收到文件，已取消。").send()
        return

    csv_files = [file for file in files if file.name.lower().endswith(".csv")]
    if len(csv_files) != 1:
        await cl.Message(content="请上传且只上传一个 CSV 文件。").send()
        return
    uploaded = csv_files[0]
    if uploaded.path is None:
        await cl.Message(content="CSV 文件尚未落地到本地，无法读取。").send()
        return

    doc_files = [
        file for file in files if file.name.lower().endswith((".md", ".txt"))
    ]
    if len(doc_files) > 1:
        await cl.Message(content="最多上传一个数据说明文档。").send()
        return
    doc_entries: tuple[DictionaryEntry, ...] = ()
    if doc_files:
        doc_file = doc_files[0]
        if doc_file.path is None:
            await cl.Message(content="数据说明文档尚未落地到本地，无法读取。").send()
            return
        try:
            doc_entries = load_dictionary_file(doc_file.path, source=doc_file.name)
        except (OSError, ValueError) as exc:
            await cl.Message(content=f"读取数据说明文档失败：{exc}").send()
            return

    try:
        connector = CsvConnector(uploaded.path, view_name=uploaded.name)
    except ValueError as exc:
        await cl.Message(content=f"读取 CSV 失败：{exc}").send()
        return

    engine = DuckDBEngine(
        memory_limit=settings.duckdb_memory_limit,
        threads=settings.duckdb_threads,
        timeout_seconds=settings.sql_timeout_seconds,
    )
    llm = build_llm(
        settings,
        response=build_mock_plan_response(connector.view_name) if settings.is_mock else None,
        ledger=_usage_ledger,
    )
    ambiguity_judge = None if settings.is_mock else LLMAmbiguityJudge(llm)
    orchestrator = Orchestrator(
        connector,
        engine,
        Planner(llm),
        clarify_gate=ClarifyGate(ambiguity_judge=ambiguity_judge),
    )
    generated_entries = build_data_dictionary(
        orchestrator.dataframe,
        profile=orchestrator.profile,
        source=uploaded.name,
    )
    dictionary_entries = (*generated_entries, *doc_entries)
    orchestrator.attach_dictionary(DictionaryQA(DataDictionaryRetriever(dictionary_entries)))
    cl.user_session.set("orchestrator", orchestrator)
    cl.user_session.set("turns", SessionTurnCounter(settings.session_turn_limit))

    frame = orchestrator.dataframe
    mode = "mock（未配置真实模型）" if settings.is_mock else settings.default_model
    await cl.Message(
        content=(
            f"已加载 `{uploaded.name}`：{len(frame)} 行 × {len(frame.columns)} 列。\n"
            f"当前模型：{mode}。\n"
            f"数据字典：{len(dictionary_entries)} 条"
            f"（文档 {len(doc_entries)} 条）。\n\n"
            f"{format_profile_report(orchestrator.profile)}"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    orchestrator: Orchestrator | None = cl.user_session.get("orchestrator")
    if orchestrator is None:
        await cl.Message(content="请先上传 CSV 文件。").send()
        return

    settings = get_settings()
    turns: SessionTurnCounter | None = cl.user_session.get("turns")
    if turns is None:
        turns = SessionTurnCounter(settings.session_turn_limit)
        cl.user_session.set("turns", turns)
    if not turns.try_consume():
        await cl.Message(
            content=(f"本会话提问次数已达上限（{turns.limit} 次），请开启新会话后再试。")
        ).send()
        return

    result: Answer | Clarification | DictionaryAnswer
    try:
        result = orchestrator.ask(message.content)
        while isinstance(result, Clarification):
            await _show_trace(orchestrator.last_trace)
            response = await cl.AskActionMessage(
                content=(f"{result.request.question}\n\n依据：{result.request.reason}"),
                actions=[
                    cl.Action(
                        name=option.value,
                        payload={"value": option.value},
                        label=option.label,
                        tooltip=option.description,
                    )
                    for option in result.request.options
                ],
            ).send()
            selected = _selected_value(response)
            if selected is None:
                await cl.Message(content="未收到有效选择，已取消本次分析。").send()
                return
            result = orchestrator.resolve(selected)
    except QuotaExceededError:
        await _show_trace(orchestrator.last_trace)
        await cl.Message(content="今日模型额度已用完，请明天再试，或改用你自己的 API key。").send()
        return
    except RepairFailed as exc:
        await _show_trace(orchestrator.last_trace)
        await cl.Message(
            content=(
                f"分析失败：已尝试 {exc.attempts} 次仍无法执行。\n"
                f"最后错误：{exc.last_error}"
            )
        ).send()
        return
    except (LLMError, QueryTimeoutError, ValueError, RuntimeError, duckdb.Error) as exc:
        await _show_trace(orchestrator.last_trace)
        await cl.Message(content=f"分析失败：{exc}").send()
        return

    await _show_trace(orchestrator.last_trace)
    elements = _trace_elements(orchestrator.last_trace)
    if isinstance(result, DictionaryAnswer):
        await cl.Message(content=result.text, elements=elements).send()
        return

    assert isinstance(result, Answer)
    await cl.Message(
        content=(
            "**生成的 SQL**\n"
            f"```sql\n{result.sql}\n```\n\n"
            "**结果**\n"
            f"```\n{format_result_table(result.dataframe)}\n```"
        ),
        elements=elements,
    ).send()


async def _show_trace(trace: Trace | None) -> None:
    """按 trace 渲染嵌套步骤时间线；详细面板在回答末尾的侧栏展示。"""
    # DEBT-020: 步骤在整轮分析完成后一次性渲染, 暂不提供 token 级实时反馈。
    if trace is None or not trace.steps:
        return
    async with cl.Step(name="分析过程", type="tool", auto_collapse=True) as parent:
        parent.output = f"共 {len(trace.steps)} 步"
        for item in trace.steps:
            async with cl.Step(name=item.name, type="tool") as step:
                step.output = item.detail or item.summary


def _trace_elements(trace: Trace | None) -> list[Any]:
    if trace is None or not trace.steps:
        return []
    return [
        cl.Text(
            name="透明面板",
            content=trace.to_panel_markdown(),
            display="side",
        )
    ]


def _selected_value(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    direct = response.get("value")
    if isinstance(direct, str):
        return direct
    payload = response.get("payload")
    if isinstance(payload, dict):
        inner = payload.get("value")
        if isinstance(inner, str):
            return inner
    return None
