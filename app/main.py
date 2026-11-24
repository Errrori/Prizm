"""Chainlit 入口：上传 CSV/Excel/SQLite → 注册 DuckDB → 体检 → 澄清门禁 → 提问 → 展示结果。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import chainlit as cl
import duckdb
from app.streaming import run_streamed

from prizm.agent.clarify import ClarifyGate, LLMAmbiguityJudge
from prizm.agent.composer import ConclusionComposer
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
from prizm.datasource.base import BaseConnector
from prizm.datasource.csv_connector import CsvConnector
from prizm.datasource.excel_connector import ExcelConnector, list_excel_sheets
from prizm.datasource.sqlite_connector import SqliteConnector, list_sqlite_tables
from prizm.engine.duckdb_engine import DuckDBEngine, QueryTimeoutError
from prizm.llm.adapter import LLMError, build_llm
from prizm.profiling.profiler import format_profile_report
from prizm.rag.ingest import DictionaryEntry, build_data_dictionary, load_dictionary_file
from prizm.rag.retriever import DataDictionaryRetriever, DictionaryAnswer, DictionaryQA
from prizm.utils.tokens import QuotaExceededError, SessionTurnCounter, SqliteUsageLedger

_usage_ledger = SqliteUsageLedger(get_settings().usage_ledger_path)


@cl.on_chat_start
async def on_chat_start() -> None:
    settings = get_settings()
    files = await cl.AskFileMessage(
        content=(
            "请上传一个数据文件（CSV / Excel .xlsx / SQLite），"
            "可选同时上传 .md/.txt 数据说明文档。"
        ),
        accept={
            "text/csv": [".csv"],
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
            "application/octet-stream": [".sqlite", ".sqlite3", ".db"],
            "text/markdown": [".md"],
            "text/plain": [".txt"],
        },
        max_files=2,
        max_size_mb=50,
    ).send()
    if not files:
        await cl.Message(content="未收到文件，已取消。").send()
        return

    data_files = [
        file
        for file in files
        if file.name.lower().endswith((".csv", ".xlsx", ".sqlite", ".sqlite3", ".db"))
    ]
    if len(data_files) != 1:
        await cl.Message(
            content="请上传且只上传一个数据文件（CSV / Excel .xlsx / SQLite）。"
        ).send()
        return
    uploaded = data_files[0]
    if uploaded.path is None:
        await cl.Message(content="数据文件尚未落地到本地，无法读取。").send()
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
        selection = await _build_connector(uploaded)
    except (OSError, ValueError) as exc:
        await cl.Message(content=f"读取数据源失败：{exc}").send()
        return
    if selection is None:
        await cl.Message(content="未选择数据表，已取消。").send()
        return
    connector, source_label = selection

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
        conclusion_composer=None if settings.is_mock else ConclusionComposer(llm),
    )
    generated_entries = build_data_dictionary(
        orchestrator.dataframe,
        profile=orchestrator.profile,
        source=source_label,
    )
    dictionary_entries = (*generated_entries, *doc_entries)
    orchestrator.attach_dictionary(DictionaryQA(DataDictionaryRetriever(dictionary_entries)))
    cl.user_session.set("orchestrator", orchestrator)
    cl.user_session.set("turns", SessionTurnCounter(settings.session_turn_limit))

    frame = orchestrator.dataframe
    mode = "mock（未配置真实模型）" if settings.is_mock else settings.default_model
    await cl.Message(
        content=(
            f"已加载 `{source_label}`：{len(frame)} 行 × {len(frame.columns)} 列。\n"
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
        await cl.Message(content="请先上传数据文件。").send()
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
    streamed: cl.Message | None = None
    try:
        result, streamed = await run_streamed(orchestrator.ask, message.content)
        while isinstance(result, Clarification):
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
            result, streamed = await run_streamed(orchestrator.resolve, selected)
    except QuotaExceededError:
        await cl.Message(content="今日模型额度已用完，请明天再试，或改用你自己的 API key。").send()
        return
    except RepairFailed as exc:
        await cl.Message(
            content=(
                f"分析失败：已尝试 {exc.attempts} 次仍无法执行。\n"
                f"最后错误：{exc.last_error}"
            )
        ).send()
        return
    except (LLMError, QueryTimeoutError, ValueError, RuntimeError, duckdb.Error) as exc:
        await cl.Message(content=f"分析失败：{exc}").send()
        return

    elements = _trace_elements(orchestrator.last_trace)
    if isinstance(result, DictionaryAnswer):
        await cl.Message(content=result.text, elements=elements).send()
        return

    assert isinstance(result, Answer)
    if result.chart_png is not None:
        elements.append(cl.Image(content=result.chart_png, name="chart.png", display="inline"))
    header = ""
    if streamed is None:
        header = f"**结论**\n{result.conclusion or '（本次未生成结论）'}\n\n"
    await cl.Message(
        content=(
            f"{header}"
            "**生成的 SQL**\n"
            f"```sql\n{result.sql}\n```\n\n"
            "**结果**\n"
            f"```\n{format_result_table(result.dataframe)}\n```"
        ),
        elements=elements,
    ).send()


async def _build_connector(uploaded: Any) -> tuple[BaseConnector, str] | None:
    """按后缀构建 connector；Excel 多 sheet / SQLite 多表时让用户选择。"""
    if uploaded.path is None:
        raise ValueError("文件尚未落地到本地，无法读取")
    path = uploaded.path
    name = uploaded.name
    lower = name.lower()

    if lower.endswith(".csv"):
        return CsvConnector(path, view_name=name), name

    if lower.endswith(".xlsx"):
        sheets = list_excel_sheets(path)
        if not sheets:
            raise ValueError(f"Excel 文件没有工作表: {name}")
        sheet = sheets[0]
        if len(sheets) > 1:
            chosen = await _ask_choice(
                f"`{name}` 有 {len(sheets)} 个工作表，请选择一个：",
                [(item, item) for item in sheets],
            )
            if chosen is None:
                return None
            sheet = chosen
        return ExcelConnector(path, view_name=name, sheet_name=sheet), (
            f"{name}（工作表：{sheet}）"
        )

    if lower.endswith((".sqlite", ".sqlite3", ".db")):
        tables = list_sqlite_tables(path)
        if not tables:
            raise ValueError(f"SQLite 文件没有可读取的数据表: {name}")
        table = tables[0]
        if len(tables) > 1:
            chosen = await _ask_choice(
                f"`{name}` 有 {len(tables)} 张表，请选择一张：",
                [(item, item) for item in tables],
            )
            if chosen is None:
                return None
            table = chosen
        return SqliteConnector(path, view_name=table, table=table), (
            f"{name}（数据表：{table}）"
        )

    if lower.endswith(".xls"):
        raise ValueError("暂不支持旧版 .xls，请另存为 .xlsx 后重试")
    raise ValueError(f"不支持的文件类型: {name}")


async def _ask_choice(title: str, options: Sequence[tuple[str, str]]) -> str | None:
    response = await cl.AskActionMessage(
        content=title,
        actions=[
            cl.Action(name=value, payload={"value": value}, label=label)
            for value, label in options
        ],
    ).send()
    return _selected_value(response)


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
