import json
from pathlib import Path

import pandas as pd
import pytest

from prizm.profiling.models import ProfileReport
from prizm.profiling.profiler import (
    ProfileOptions,
    format_profile_prompt,
    format_profile_report,
    profile_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def clean_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5],
            "channel": ["线上", "线上", "线下", "线上", "线下"],
            "quantity": [1, 2, 3, 4, 5],
            "sales": [100, 200, 300, 400, 500],
        }
    )


def test_clean_frame_has_no_anomalies() -> None:
    report = profile_frame(clean_frame())

    assert report.row_count == 5
    assert report.column_count == 4
    assert report.has_anomalies is False
    assert report.duplicates.key_columns == ("order_id",)
    assert len(report.sample_rows) == 5


def test_missing_values_count_blank_strings() -> None:
    frame = pd.DataFrame({"value": ["", "   ", "x", None]})

    report = profile_frame(frame)

    missing = [anomaly for anomaly in report.anomalies if anomaly.code == "missing_values"]
    assert len(missing) == 1
    assert missing[0].column == "value"
    assert missing[0].count == 3
    assert report.columns[0].missing_count == 3


def test_exact_and_primary_key_duplicates_reported_separately() -> None:
    frame = pd.DataFrame(
        {
            "order_id": [1, 1, 2],
            "sales": [10, 10, 20],
        }
    )

    report = profile_frame(frame)

    codes = report.anomaly_codes
    assert "exact_duplicate_rows" in codes
    assert "duplicate_primary_key" in codes
    assert report.duplicates.exact_duplicate_rows == 1
    assert report.duplicates.key_duplicate_group_count == 1
    assert report.duplicates.key_duplicate_row_count == 2


def test_primary_key_can_be_overridden() -> None:
    frame = pd.DataFrame(
        {
            "order_id": [1, 2, 3],
            "code": ["a", "a", "b"],
        }
    )

    report = profile_frame(frame, primary_key="code")

    assert report.duplicates.key_columns == ("code",)
    assert report.duplicates.key_duplicate_group_count == 1


def test_missing_primary_key_column_raises() -> None:
    with pytest.raises(ValueError, match="主键列不存在"):
        profile_frame(clean_frame(), primary_key="missing")


def test_dirty_numeric_formats_detected() -> None:
    frame = pd.DataFrame({"sales": ["1,200", "¥800", "3万", "50%"]})

    report = profile_frame(frame)

    dirty = [anomaly for anomaly in report.anomalies if anomaly.code == "dirty_numeric_format"]
    assert len(dirty) == 1
    assert dirty[0].column == "sales"
    assert dirty[0].count == 4
    assert report.columns[0].semantic_type == "numeric"


def test_plain_numeric_strings_are_not_dirty() -> None:
    frame = pd.DataFrame({"sales": ["100", "200", "300"]})

    report = profile_frame(frame)

    assert "dirty_numeric_format" not in report.anomaly_codes


def test_text_with_units_is_not_dirty_numeric() -> None:
    frame = pd.DataFrame({"channel": ["百货", "超市", "百货", "超市"]})

    report = profile_frame(frame)

    assert "dirty_numeric_format" not in report.anomaly_codes


def test_numeric_stats_summarize_distribution() -> None:
    report = profile_frame(clean_frame())
    sales = next(column for column in report.columns if column.name == "sales")

    assert sales.numeric_stats is not None
    assert sales.numeric_stats.min == 100
    assert sales.numeric_stats.max == 500
    assert sales.numeric_stats.mean == 300


def test_sample_rows_are_capped_and_prompt_excludes_full_data() -> None:
    frame = pd.DataFrame({"id": range(100), "value": [f"v{i}" for i in range(100)]})

    report = profile_frame(frame, options=ProfileOptions(sample_rows=5))
    prompt = format_profile_prompt(report)

    assert len(report.sample_rows) == 5
    assert "v0" in prompt
    assert "v4" in prompt
    assert "v5" not in prompt
    assert "v99" not in prompt


def test_prompt_respects_max_chars() -> None:
    frame = pd.DataFrame({f"column_{i}": [f"value_{i}_{j}" for j in range(20)] for i in range(20)})

    report = profile_frame(frame)
    prompt = format_profile_prompt(report, max_chars=300)

    assert len(prompt) <= 300


def test_report_formatter_contains_anomalies() -> None:
    frame = pd.DataFrame({"order_id": [1, 1], "sales": ["1,200", "1,200"]})

    text = format_profile_report(profile_frame(frame))

    assert "完全重复" in text
    assert "主键" in text
    assert "千分位" in text


def test_empty_frame_is_supported() -> None:
    report = profile_frame(pd.DataFrame({"a": pd.Series(dtype="object")}))

    assert report.row_count == 0
    assert report.sample_rows == ()
    assert report.anomalies == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rows": -1},
        {"top_values": -1},
        {"max_examples": -1},
        {"dirty_ratio_threshold": 0},
        {"max_prompt_chars": 100},
    ],
)
def test_invalid_profile_options_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ProfileOptions(**kwargs)


def test_dirty_orders_dataset_detects_all_seeded_anomalies() -> None:
    frame = pd.read_csv(REPO_ROOT / "eval" / "datasets" / "orders_dirty.csv")

    report = profile_frame(frame)

    assert report.anomaly_codes >= {
        "missing_values",
        "exact_duplicate_rows",
        "duplicate_primary_key",
        "dirty_numeric_format",
    }
    assert isinstance(report, ProfileReport)
    assert report.duplicates.exact_duplicate_rows == 1
    assert report.duplicates.key_columns == ("order_id",)
    assert report.duplicates.key_duplicate_group_count == 2
    missing_columns = {
        anomaly.column for anomaly in report.anomalies if anomaly.code == "missing_values"
    }
    assert missing_columns == {"quantity", "status"}
    dirty = [anomaly for anomaly in report.anomalies if anomaly.code == "dirty_numeric_format"]
    assert [(anomaly.column, anomaly.count) for anomaly in dirty] == [("sales", 1)]
    assert len(report.anomalies) == 5


def test_sample_rows_are_json_serializable() -> None:
    frame = pd.DataFrame({"when": pd.to_datetime(["2026-01-01"]), "value": [1.5]})

    report = profile_frame(frame)

    assert json.loads(json.dumps(report.sample_rows))[0]["when"] == "2026-01-01T00:00:00"
