"""Chainlit 入口：上传 CSV → 注册 DuckDB → 提问 → 展示 SQL 与结果。

阶段 1 最薄链路：不含数据体检、澄清、自修复与图表。
"""

from __future__ import annotations

import chainlit as cl
import duckdb

from prizm.agent.orchestrator import Orchestrator, format_result_table
from prizm.agent.planner import Planner, build_mock_plan_response
from prizm.config import get_settings
from prizm.datasource.csv_connector import CsvConnector
from prizm.engine.duckdb_engine import DuckDBEngine, QueryTimeoutError
from prizm.llm.adapter import LLMError, build_llm
from prizm.utils.tokens import DailyUsageLedger, QuotaExceededError, SessionTurnCounter

_usage_ledger = DailyUsageLedger()


@cl.on_chat_start
async def on_chat_start() -> None:
    settings = get_settings()
    files = await cl.AskFileMessage(
        content="请上传一个 CSV 文件（阶段 1 仅支持 CSV）。",
        accept=["text/csv"],
        max_files=1,
        max_size_mb=50,
    ).send()
    if not files:
        await cl.Message(content="未收到文件，已取消。").send()
        return

    uploaded = files[0]
    if uploaded.path is None:
        await cl.Message(content="文件尚未落地到本地，无法读取。").send()
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
    orchestrator = Orchestrator(connector, engine, Planner(llm))
    cl.user_session.set("orchestrator", orchestrator)
    cl.user_session.set("turns", SessionTurnCounter(settings.session_turn_limit))

    frame = orchestrator.dataframe
    mode = "mock（未配置真实模型）" if settings.is_mock else settings.default_model
    await cl.Message(
        content=(
            f"已加载 `{uploaded.name}`：{len(frame)} 行 × {len(frame.columns)} 列。\n"
            f"当前模型：{mode}。可以开始提问。"
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
            content=(
                f"本会话提问次数已达上限（{turns.limit} 次），"
                "请开启新会话后再试。"
            )
        ).send()
        return

    try:
        async with cl.Step(name="规划与执行", type="tool") as step:
            step.input = message.content
            answer = orchestrator.ask(message.content)
            step.output = answer.sql
    except QuotaExceededError:
        await cl.Message(
            content="今日模型额度已用完，请明天再试，或改用你自己的 API key。"
        ).send()
        return
    except (LLMError, QueryTimeoutError, ValueError, duckdb.Error) as exc:
        await cl.Message(content=f"分析失败：{exc}").send()
        return

    await cl.Message(
        content=(
            "**生成的 SQL**\n"
            f"```sql\n{answer.sql}\n```\n\n"
            "**结果**\n"
            f"```\n{format_result_table(answer.dataframe)}\n```"
        )
    ).send()
