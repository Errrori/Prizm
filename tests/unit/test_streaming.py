import sys
import threading
import types
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd
import pytest

from prizm.agent.orchestrator import Answer
from prizm.agent.progress import ProgressEvent


class FakeStep:
    instances: ClassVar[list["FakeStep"]] = []

    def __init__(
        self,
        name: str = "Assistant",
        type: str = "undefined",
        id: str | None = None,
        parent_id: str | None = None,
        **_: Any,
    ) -> None:
        self.name = name
        self.type = type
        self.id = id or f"step-{len(FakeStep.instances)}"
        self.parent_id = parent_id
        self.output = ""
        self.sent = False
        self.updates = 0
        FakeStep.instances.append(self)

    async def send(self) -> None:
        self.sent = True

    async def update(self) -> None:
        self.updates += 1


class FakeMessage:
    def __init__(self, content: str = "") -> None:
        self.content = content
        self.tokens: list[str] = []
        self.sent = False
        self.updates = 0

    async def send(self) -> None:
        self.sent = True

    async def stream_token(self, token: str, *_: Any, **__: Any) -> None:
        self.tokens.append(token)

    async def update(self) -> None:
        self.updates += 1


class FakeChainlit:
    Step = FakeStep
    Message = FakeMessage


def _install_chainlit_stub() -> None:
    """预置 chainlit 桩，避免真实 chainlit 导入时 load_dotenv 污染 os.environ。"""
    if "chainlit" not in sys.modules:
        module = types.ModuleType("chainlit")
        module.Step = FakeStep  # type: ignore[attr-defined]
        module.Message = FakeMessage  # type: ignore[attr-defined]
        sys.modules["chainlit"] = module


_install_chainlit_stub()

import app.streaming as streaming  # noqa: E402


@dataclass
class _Run:
    events: list[ProgressEvent] = field(default_factory=list)

    def __call__(self, *, progress: Any) -> Answer:
        progress(ProgressEvent(kind="step_start", name="SQL 规划", summary="正在生成…"))
        progress(
            ProgressEvent(
                kind="step",
                name="SQL 规划",
                status="ok",
                summary="已生成只读 SQL",
                detail="SELECT 1",
            )
        )
        progress(
            ProgressEvent(kind="step_start", name="结论生成", summary="正在归纳…")
        )
        progress(ProgressEvent(kind="conclusion_token", text="销售额"))
        progress(ProgressEvent(kind="conclusion_token", text="合计 300。"))
        progress(
            ProgressEvent(
                kind="step",
                name="结论生成",
                status="ok",
                summary="已用模型归纳结果",
                detail="销售额合计 300。",
            )
        )
        return Answer(
            question="q",
            sql="SELECT 1",
            dataframe=pd.DataFrame({"total": [300]}),
            conclusion="销售额合计 300。",
            conclusion_source="llm",
        )


@pytest.fixture(autouse=True)
def _reset_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeStep.instances.clear()
    monkeypatch.setattr(streaming, "cl", FakeChainlit)


@pytest.mark.asyncio
async def test_run_streamed_creates_and_updates_single_step_per_name() -> None:
    result, message = await streaming.run_streamed(_Run())

    assert isinstance(result, Answer)
    assert result.conclusion == "销售额合计 300。"

    names = [step.name for step in FakeStep.instances]
    assert names.count("SQL 规划") == 1
    assert names.count("结论生成") == 1
    assert names[0] == "分析过程"

    sql_step = next(step for step in FakeStep.instances if step.name == "SQL 规划")
    assert sql_step.output == "SELECT 1"
    assert sql_step.updates == 2

    assert message is not None
    assert "".join(message.tokens) == "销售额合计 300。"
    assert message.content == "**结论**\n销售额合计 300。"


@pytest.mark.asyncio
async def test_run_streamed_reconciles_template_fallback() -> None:
    def call(*, progress: Any) -> Answer:
        progress(ProgressEvent(kind="conclusion_token", text="半截模型结论"))
        return Answer(
            question="q",
            sql="SELECT 1",
            dataframe=pd.DataFrame(),
            conclusion="查询结果为空，未返回任何数据行。",
            conclusion_source="template",
        )

    _, message = await streaming.run_streamed(call)

    assert message is not None
    assert message.content == "**结论**\n查询结果为空，未返回任何数据行。"


@pytest.mark.asyncio
async def test_run_streamed_runs_call_in_worker_thread() -> None:
    seen: list[int] = []

    def call(*, progress: Any) -> Answer:
        seen.append(threading.get_ident())
        return Answer(question="q", sql="SELECT 1", dataframe=pd.DataFrame())

    await streaming.run_streamed(call)

    assert seen
    assert seen[0] != threading.main_thread().ident
