import pytest

from prizm.agent.memory import ConversationMemory, Turn


def make_turn(index: int) -> Turn:
    return Turn(
        question=f"问题 {index}",
        sql=f"SELECT {index}",
        columns=("value",),
        row_count=index,
        preview=({"value": index},),
    )


def test_turn_rejects_empty_question_and_sql() -> None:
    with pytest.raises(ValueError, match="question"):
        Turn(question="  ", sql="SELECT 1")
    with pytest.raises(ValueError, match="sql"):
        Turn(question="问题", sql="  ")


def test_turn_rejects_negative_row_count() -> None:
    with pytest.raises(ValueError, match="row_count"):
        Turn(question="问题", sql="SELECT 1", row_count=-1)


def test_memory_keeps_only_recent_turns() -> None:
    memory = ConversationMemory(max_turns=2)

    for index in range(1, 4):
        memory.add(make_turn(index))

    assert len(memory) == 2
    assert [turn.question for turn in memory.turns] == ["问题 2", "问题 3"]


def test_memory_formats_turn_for_prompt() -> None:
    memory = ConversationMemory()
    memory.add(make_turn(1))

    text = memory.format_for_prompt()

    assert text is not None
    assert "问题 1" in text
    assert "SELECT 1" in text
    assert "value" in text
    assert "1 行" in text


def test_empty_memory_has_no_prompt_fragment() -> None:
    assert ConversationMemory().format_for_prompt() is None


def test_prompt_truncation_keeps_newest_turn() -> None:
    memory = ConversationMemory(max_turns=2)
    memory.add(make_turn(1))
    memory.add(make_turn(2))

    text = memory.format_for_prompt(max_chars=30)

    assert text is not None
    assert text.endswith("…")
    assert "问题 2" in text
    assert "问题 1" not in text


def test_preview_is_truncated_in_prompt() -> None:
    memory = ConversationMemory()
    memory.add(
        Turn(
            question="问题",
            sql="SELECT 1",
            columns=("payload",),
            row_count=1,
            preview=({"payload": "x" * 2_000},),
        )
    )

    text = memory.format_for_prompt()

    assert text is not None
    assert "…" in text
    assert "x" * 2_000 not in text


def test_clear_and_validation() -> None:
    with pytest.raises(ValueError, match="max_turns"):
        ConversationMemory(max_turns=0)
    with pytest.raises(ValueError, match="prompt_max_chars"):
        ConversationMemory(prompt_max_chars=0)
    with pytest.raises(ValueError, match="max_chars"):
        ConversationMemory().format_for_prompt(max_chars=0)

    memory = ConversationMemory()
    memory.add(make_turn(1))
    memory.clear()
    assert memory.turns == ()
