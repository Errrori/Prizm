"""把同步编排的进度事件桥接到 Chainlit：实时步骤 + 结论 token。

编排在独立线程中运行，事件经 ``asyncio.Queue`` 回到事件循环渲染，
避免阻塞 UI，也不改变编排本身的同步语义。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import chainlit as cl

from prizm.agent.orchestrator import Answer, Clarification, DictionaryAnswer
from prizm.agent.progress import ProgressEvent

TurnResult = Answer | Clarification | DictionaryAnswer


async def run_streamed(
    call: Callable[..., TurnResult],
    *args: Any,
) -> tuple[TurnResult, cl.Message | None]:
    """运行一次同步编排并实时渲染进度。

    - ``step_start`` 创建一个“进行中”的步骤；
    - ``step`` 按名称更新同一个步骤的最终结果；
    - ``conclusion_token`` 逐字写入结论消息。

    返回最终结果与流式结论消息（无 token 时为 None），
    以便编排结束后用权威文本校正结论（模型失败回退模板时尤其需要）。
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[ProgressEvent | None] = asyncio.Queue()

    def emit(event: ProgressEvent) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    parent = cl.Step(name="分析过程", type="tool", auto_collapse=True)
    await parent.send()
    rendered: dict[str, cl.Step] = {}
    conclusion_msg: cl.Message | None = None

    async def consume() -> None:
        nonlocal conclusion_msg
        while True:
            event = await queue.get()
            if event is None:
                return
            if event.kind == "step_start":
                step = cl.Step(name=event.name, type="tool", parent_id=parent.id)
                await step.send()
                step.output = (
                    f"{event.summary}（进行中…）" if event.summary else "进行中…"
                )
                await step.update()
                rendered[event.name] = step
            elif event.kind == "step":
                existing = rendered.get(event.name)
                if existing is None:
                    existing = cl.Step(
                        name=event.name, type="tool", parent_id=parent.id
                    )
                    await existing.send()
                    rendered[event.name] = existing
                existing.output = event.detail or event.summary
                await existing.update()
            elif event.kind == "conclusion_token":
                if conclusion_msg is None:
                    conclusion_msg = cl.Message(content="**结论**\n")
                    await conclusion_msg.send()
                await conclusion_msg.stream_token(event.text)

    consumer = asyncio.create_task(consume())
    try:
        result = await asyncio.to_thread(call, *args, progress=emit)
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, None)
        await consumer

    parent.output = f"共 {len(rendered)} 步"
    await parent.update()
    if conclusion_msg is not None and isinstance(result, Answer):
        conclusion_msg.content = (
            f"**结论**\n{result.conclusion or '（本次未生成结论）'}"
        )
        await conclusion_msg.update()
    return result, conclusion_msg
