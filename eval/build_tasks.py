"""独立脚本生成评测任务与标准答案。

标准答案全部由与 Agent SQL 路径无关的 pandas 计算得出，生成后写入
``eval/tasks.jsonl``。修改数据集或任务模板后重新运行本脚本即可复现；
任务文件不应手工改动标准答案。

用法：
    uv run python -m eval.build_tasks
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

DATASETS = Path(__file__).resolve().parent / "datasets"
TASKS_PATH = Path(__file__).resolve().parent / "tasks.jsonl"

ORDERS = "orders.csv"
ORDERS_DIRTY = "orders_dirty.csv"
MONTHLY = "monthly_sales.csv"
MONTHLY_DIRTY = "monthly_sales_dirty.csv"
TRAFFIC = "channel_traffic.csv"
TRAFFIC_DIRTY = "channel_traffic_dirty.csv"


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(
        list[dict[str, Any]],
        frame.reset_index(drop=True).to_dict(orient="records"),
    )


def rank_sum(df: pd.DataFrame, key: str, value: str) -> list[dict[str, Any]]:
    out = df.groupby(key, as_index=False)[value].sum()
    return _records(out.sort_values(value, ascending=False))


def trend_sum(df: pd.DataFrame, key: str, value: str) -> list[dict[str, Any]]:
    out = df.groupby(key, as_index=False)[value].sum()
    return _records(out.sort_values(key, ascending=True))


def rank_count(df: pd.DataFrame, key: str) -> list[dict[str, Any]]:
    out = df.groupby(key, as_index=False).size().rename(columns={"size": "count"})
    return _records(out.sort_values(["count", key], ascending=[False, True]))


def trend_count(df: pd.DataFrame, key: str) -> list[dict[str, Any]]:
    out = df.groupby(key, as_index=False).size().rename(columns={"size": "count"})
    return _records(out.sort_values(key, ascending=True))


def trend_mean(df: pd.DataFrame, key: str, value: str) -> list[dict[str, Any]]:
    out = df.groupby(key, as_index=False)[value].mean()
    return _records(out.sort_values(key, ascending=True))


def top_key(df: pd.DataFrame, key: str, value: str) -> str:
    totals = df.groupby(key)[value].sum().sort_values(ascending=False)
    return str(totals.index[0])


def bottom_key(df: pd.DataFrame, key: str, value: str) -> str:
    totals = df.groupby(key)[value].sum().sort_values(ascending=True)
    return str(totals.index[0])


def _task(
    task_id: str,
    category: str,
    dataset: str,
    question: str,
    expected: Any,
    *,
    session: str | None = None,
    expects_clarification: bool = False,
) -> dict[str, Any]:
    record = {
        "id": task_id,
        "category": category,
        "dataset": dataset,
        "question": question,
        "expected": expected,
        "expects_clarification": expects_clarification,
    }
    if session is not None:
        record["session"] = session
    return record


def _clarify(
    task_id: str,
    dataset: str,
    question: str,
) -> dict[str, Any]:
    return _task(
        task_id,
        "cleaning",
        dataset,
        question,
        None,
        expects_clarification=True,
    )


def build_tasks() -> list[dict[str, Any]]:
    orders = pd.read_csv(DATASETS / ORDERS)
    orders["month"] = orders["order_date"].str[:7]
    orders_done = orders[orders["status"] == "已完成"]
    orders_jan = orders[(orders["month"] == "2026-01") & (orders["status"] == "已完成")]
    orders_mar = orders[(orders["month"] == "2026-03") & (orders["status"] == "已完成")]

    monthly = pd.read_csv(DATASETS / MONTHLY)
    monthly_standard = monthly[monthly["product_line"] == "标准版"]
    monthly_pro = monthly[monthly["product_line"] == "专业版"]
    monthly_h1 = monthly[monthly["month"] <= "2025-06"]
    monthly_h2 = monthly[monthly["month"] >= "2025-07"]

    traffic = pd.read_csv(DATASETS / TRAFFIC)
    traffic_search = traffic[traffic["channel"] == "搜索"]
    traffic_social = traffic[traffic["channel"] == "社交"]
    traffic_direct = traffic[traffic["channel"] == "直邮"]
    traffic_h2 = traffic[traffic["month"] >= "2025-07"]
    traffic_top2 = traffic[traffic["channel"].isin(["搜索", "社交"])]

    tasks: list[dict[str, Any]] = []

    # ---- orders.csv:原始 7 题(保留 id,不修改标准答案) ----------------
    orders_top_channel = top_key(orders_done, "channel", "sales")
    tasks.extend(
        [
            _task(
                "clean-001",
                "cleaning",
                ORDERS,
                "这份订单数据里是否存在完全重复的行？如果存在，返回重复行数。",
                int(orders.duplicated().sum()),
            ),
            _task(
                "clean-002",
                "aggregation",
                ORDERS,
                "统计 2026 年第一季度状态为“已完成”的订单销售额总额。",
                int(orders_done["sales"].sum()),
            ),
            _task(
                "clean-003",
                "aggregation",
                ORDERS,
                "按渠道统计状态为“已完成”的订单销售额，并按销售额从高到低排序。",
                rank_sum(orders_done, "channel", "sales"),
            ),
            _task(
                "clean-004",
                "trend",
                ORDERS,
                "按月份统计状态为“已完成”的订单销售额趋势。",
                trend_sum(orders_done, "month", "sales"),
            ),
            _task(
                "clean-005",
                "trend",
                ORDERS,
                "哪个月的已完成订单销售额最高？只返回月份。",
                top_key(orders_done, "month", "sales"),
            ),
            _task(
                "clean-006",
                "attribution",
                ORDERS,
                "哪个渠道对已完成订单销售额贡献最大？只返回渠道名称。",
                orders_top_channel,
                session="orders-channel-followup",
            ),
            _task(
                "clean-007",
                "multi_turn",
                ORDERS,
                "承接上一题的筛选条件，只看已完成订单，按产品统计销售额，并从高到低排序。",
                rank_sum(orders_done, "product", "sales"),
                session="orders-channel-followup",
            ),
        ]
    )

    # ---- orders_dirty.csv:异常口径澄清 ---------------------------------
    tasks.extend(
        [
            _clarify(
                "dirty-001",
                ORDERS_DIRTY,
                "检测到完全重复行，也有同一订单号对应不同金额的记录。"
                "统计销售额前是否应该去重？按哪种去重口径？",
            ),
            _clarify(
                "dirty-002",
                ORDERS_DIRTY,
                "sales 列混入了千分位格式，应该先清洗为数值再统计，还是按原样处理？",
            ),
            _clarify(
                "dirty-003",
                ORDERS_DIRTY,
                "quantity 存在缺失值，这些订单应该剔除、按 0 处理，还是先确认原始数据？",
            ),
            _clarify(
                "dirty-004",
                ORDERS_DIRTY,
                "主键 order_id 存在重复记录，统计销售额前按哪种口径处理？",
            ),
        ]
    )

    # ---- orders.csv:清洗 ----------------------------------------------
    tasks.extend(
        [
            _task(
                "ord-001",
                "cleaning",
                ORDERS,
                "订单号 order_id 有多少个重复出现的订单号？",
                int(orders["order_id"].duplicated().sum()),
            ),
            _task(
                "ord-002",
                "cleaning",
                ORDERS,
                "quantity 列有多少个缺失值？",
                int(orders["quantity"].isna().sum()),
            ),
            _task(
                "ord-003",
                "cleaning",
                ORDERS,
                "status 列有多少个不同取值？",
                int(orders["status"].nunique()),
            ),
            _task(
                "ord-004",
                "cleaning",
                ORDERS,
                "channel 列有多少个不同取值？",
                int(orders["channel"].nunique()),
            ),
            _task(
                "ord-005",
                "cleaning",
                ORDERS,
                "region 列有多少个不同取值？",
                int(orders["region"].nunique()),
            ),
        ]
    )

    # ---- orders.csv:聚合 ----------------------------------------------
    tasks.extend(
        [
            _task(
                "ord-006",
                "aggregation",
                ORDERS,
                "全部订单的销售额总额是多少？",
                int(orders["sales"].sum()),
            ),
            _task(
                "ord-007",
                "aggregation",
                ORDERS,
                "状态为“已完成”的订单有多少条？",
                len(orders_done),
            ),
            _task(
                "ord-008",
                "aggregation",
                ORDERS,
                "状态为“已取消”的订单有多少条？",
                len(orders[orders["status"] == "已取消"]),
            ),
            _task(
                "ord-009",
                "aggregation",
                ORDERS,
                "全部订单的 sales 平均值是多少？",
                float(orders["sales"].mean()),
            ),
            _task(
                "ord-010",
                "aggregation",
                ORDERS,
                "已完成订单的 quantity 总和是多少？",
                int(orders_done["quantity"].sum()),
            ),
            _task(
                "ord-011",
                "aggregation",
                ORDERS,
                "已完成订单中单笔销售额最大是多少？",
                int(orders_done["sales"].max()),
            ),
            _task(
                "ord-012",
                "aggregation",
                ORDERS,
                "已完成订单中单笔销售额最小是多少？",
                int(orders_done["sales"].min()),
            ),
            _task(
                "ord-013",
                "aggregation",
                ORDERS,
                "线下渠道的订单有多少条？",
                len(orders[orders["channel"] == "线下"]),
            ),
            _task(
                "ord-014",
                "aggregation",
                ORDERS,
                "线上渠道已完成订单的销售额总额是多少？",
                int(orders_done[orders_done["channel"] == "线上"]["sales"].sum()),
            ),
            _task(
                "ord-015",
                "aggregation",
                ORDERS,
                "分销渠道已完成订单的销售额总额是多少？",
                int(orders_done[orders_done["channel"] == "分销"]["sales"].sum()),
            ),
            _task(
                "ord-016",
                "aggregation",
                ORDERS,
                "华东地区已完成订单的销售额总额是多少？",
                int(orders_done[orders_done["region"] == "华东"]["sales"].sum()),
            ),
            _task(
                "ord-017",
                "aggregation",
                ORDERS,
                "旗舰版已完成订单的销售额总额是多少？",
                int(orders_done[orders_done["product"] == "旗舰版"]["sales"].sum()),
            ),
        ]
    )

    # ---- orders.csv:趋势 ----------------------------------------------
    tasks.extend(
        [
            _task(
                "ord-018",
                "trend",
                ORDERS,
                "哪个月的已完成订单销售额最低？只返回月份。",
                bottom_key(orders_done, "month", "sales"),
            ),
            _task(
                "ord-019",
                "trend",
                ORDERS,
                "按月份统计已完成订单数量，并按月份升序排列。",
                trend_count(orders_done, "month"),
            ),
            _task(
                "ord-020",
                "trend",
                ORDERS,
                "按月份统计全部订单销售额，并按月份升序排列。",
                trend_sum(orders, "month", "sales"),
            ),
            _task(
                "ord-021",
                "trend",
                ORDERS,
                "2026 年 3 月已完成订单销售额比 1 月多多少？",
                int(orders_mar["sales"].sum() - orders_jan["sales"].sum()),
            ),
            _task(
                "ord-022",
                "trend",
                ORDERS,
                "按月份统计分销渠道已完成订单销售额，并按月份升序排列。",
                trend_sum(orders_done[orders_done["channel"] == "分销"], "month", "sales"),
            ),
            _task(
                "ord-023",
                "trend",
                ORDERS,
                "按月份统计线上渠道已完成订单数量，并按月份升序排列。",
                trend_count(orders_done[orders_done["channel"] == "线上"], "month"),
            ),
            _task(
                "ord-024",
                "trend",
                ORDERS,
                "哪个月的订单数量最多？只返回月份。",
                top_key(orders, "month", "sales"),
            ),
            _task(
                "ord-025",
                "trend",
                ORDERS,
                "按月份统计已完成订单的平均销售额，并按月份升序排列。",
                trend_mean(orders_done, "month", "sales"),
            ),
        ]
    )

    # ---- orders.csv:归因 ----------------------------------------------
    tasks.extend(
        [
            _task(
                "ord-026",
                "attribution",
                ORDERS,
                "哪个产品对已完成订单销售额贡献最大？只返回产品名称。",
                top_key(orders_done, "product", "sales"),
            ),
            _task(
                "ord-027",
                "attribution",
                ORDERS,
                "按产品统计已完成订单销售额，并从高到低排序。",
                rank_sum(orders_done, "product", "sales"),
            ),
            _task(
                "ord-028",
                "attribution",
                ORDERS,
                "哪个地区对已完成订单销售额贡献最大？只返回地区名称。",
                top_key(orders_done, "region", "sales"),
            ),
            _task(
                "ord-029",
                "attribution",
                ORDERS,
                "按地区统计已完成订单销售额，并从高到低排序。",
                rank_sum(orders_done, "region", "sales"),
            ),
            _task(
                "ord-030",
                "attribution",
                ORDERS,
                "哪个渠道的已完成订单数量最多？只返回渠道名称。",
                rank_count(orders_done, "channel")[0]["channel"],
            ),
            _task(
                "ord-031",
                "attribution",
                ORDERS,
                "按渠道统计全部订单销售额，并从高到低排序。",
                rank_sum(orders, "channel", "sales"),
            ),
            _task(
                "ord-032",
                "attribution",
                ORDERS,
                "哪个产品的全部订单销售额最低？只返回产品名称。",
                bottom_key(orders, "product", "sales"),
            ),
            _task(
                "ord-033",
                "attribution",
                ORDERS,
                "分销渠道已完成订单销售额比线上渠道多多少？",
                int(
                    orders_done[orders_done["channel"] == "分销"]["sales"].sum()
                    - orders_done[orders_done["channel"] == "线上"]["sales"].sum()
                ),
            ),
            _task(
                "ord-034",
                "attribution",
                ORDERS,
                "按地区统计已完成订单数量，并从高到低排序；数量相同时按地区名称升序。",
                rank_count(orders_done, "region"),
            ),
        ]
    )

    # ---- orders.csv:多轮 ----------------------------------------------
    orders_top_region = top_key(orders_done, "region", "sales")
    tasks.extend(
        [
            _task(
                "ord-035",
                "multi_turn",
                ORDERS,
                "按渠道统计全部订单销售额，并从高到低排序。",
                rank_sum(orders, "channel", "sales"),
                session="orders-all-by-channel",
            ),
            _task(
                "ord-036",
                "multi_turn",
                ORDERS,
                "只看已完成订单，再按渠道统计销售额并从高到低排序。",
                rank_sum(orders_done, "channel", "sales"),
                session="orders-all-by-channel",
            ),
            _task(
                "ord-037",
                "multi_turn",
                ORDERS,
                "按产品统计已完成订单销售额，并从高到低排序。",
                rank_sum(orders_done, "product", "sales"),
                session="orders-done-by-product",
            ),
            _task(
                "ord-038",
                "multi_turn",
                ORDERS,
                "只看线上渠道，再按产品统计已完成订单销售额并从高到低排序。",
                rank_sum(
                    orders_done[orders_done["channel"] == "线上"], "product", "sales"
                ),
                session="orders-done-by-product",
            ),
            _task(
                "ord-039",
                "multi_turn",
                ORDERS,
                "统计 2026 年 3 月已完成订单销售额总额。",
                int(orders_mar["sales"].sum()),
                session="orders-march-done",
            ),
            _task(
                "ord-040",
                "multi_turn",
                ORDERS,
                "在此基础上，只看分销渠道的销售额。",
                int(orders_mar[orders_mar["channel"] == "分销"]["sales"].sum()),
                session="orders-march-done",
            ),
            _task(
                "ord-041",
                "multi_turn",
                ORDERS,
                "按地区统计已完成订单数量，并从高到低排序；数量相同时按地区名称升序。",
                rank_count(orders_done, "region"),
                session="orders-done-by-region",
            ),
            _task(
                "ord-042",
                "multi_turn",
                ORDERS,
                "只看华东地区的已完成订单，返回订单数量。",
                len(orders_done[orders_done["region"] == "华东"]),
                session="orders-done-by-region",
            ),
            _task(
                "ord-043",
                "multi_turn",
                ORDERS,
                "哪个地区对已完成订单销售额贡献最大？只返回地区名称。",
                orders_top_region,
                session="orders-top-region-followup",
            ),
            _task(
                "ord-044",
                "multi_turn",
                ORDERS,
                "刚才那个地区的已完成订单数量是多少？",
                len(orders_done[orders_done["region"] == orders_top_region]),
                session="orders-top-region-followup",
            ),
        ]
    )

    # ---- monthly_sales.csv:清洗 ---------------------------------------
    tasks.extend(
        [
            _task(
                "mon-001",
                "cleaning",
                MONTHLY,
                "monthly_sales 数据里有多少行完全重复？",
                int(monthly.duplicated().sum()),
            ),
            _task(
                "mon-002",
                "cleaning",
                MONTHLY,
                "sales_id 有多少个重复出现的编号？",
                int(monthly["sales_id"].duplicated().sum()),
            ),
            _task(
                "mon-003",
                "cleaning",
                MONTHLY,
                "units_sold 列有多少个缺失值？",
                int(monthly["units_sold"].isna().sum()),
            ),
            _task(
                "mon-004",
                "cleaning",
                MONTHLY,
                "product_line 列有多少个不同取值？",
                int(monthly["product_line"].nunique()),
            ),
        ]
    )

    # ---- monthly_sales_dirty.csv:异常口径澄清 -------------------------
    tasks.extend(
        [
            _clarify(
                "mon-dirty-001",
                MONTHLY_DIRTY,
                "检测到完全重复行，统计 revenue 收入前是否应该去重？按哪种去重口径？",
            ),
            _clarify(
                "mon-dirty-002",
                MONTHLY_DIRTY,
                "revenue 列混入了千分位和货币符，应该先清洗为数值再统计，还是按原样处理？",
            ),
            _clarify(
                "mon-dirty-003",
                MONTHLY_DIRTY,
                "units_sold 存在缺失值，这些记录应该剔除、按 0 处理，还是先确认原始数据？",
            ),
            _clarify(
                "mon-dirty-004",
                MONTHLY_DIRTY,
                "sales_id 主键存在重复记录，统计收入前按哪种口径处理？",
            ),
        ]
    )

    # ---- monthly_sales.csv:聚合 ---------------------------------------
    tasks.extend(
        [
            _task(
                "mon-005",
                "aggregation",
                MONTHLY,
                "全年 revenue 总额是多少？",
                int(monthly["revenue"].sum()),
            ),
            _task(
                "mon-006",
                "aggregation",
                MONTHLY,
                "标准版全年 revenue 总额是多少？",
                int(monthly_standard["revenue"].sum()),
            ),
            _task(
                "mon-007",
                "aggregation",
                MONTHLY,
                "专业版全年 revenue 总额是多少？",
                int(monthly_pro["revenue"].sum()),
            ),
            _task(
                "mon-008",
                "aggregation",
                MONTHLY,
                "全年 units_sold 总和是多少？",
                int(monthly["units_sold"].sum()),
            ),
            _task(
                "mon-009",
                "aggregation",
                MONTHLY,
                "专业版全年 units_sold 总和是多少？",
                int(monthly_pro["units_sold"].sum()),
            ),
            _task(
                "mon-010",
                "aggregation",
                MONTHLY,
                "单行 revenue 的平均值是多少？",
                float(monthly["revenue"].mean()),
            ),
            _task(
                "mon-011",
                "aggregation",
                MONTHLY,
                "华东地区全年 revenue 总额是多少？",
                int(monthly[monthly["region"] == "华东"]["revenue"].sum()),
            ),
            _task(
                "mon-012",
                "aggregation",
                MONTHLY,
                "华北地区全年 revenue 总额是多少？",
                int(monthly[monthly["region"] == "华北"]["revenue"].sum()),
            ),
            _task(
                "mon-013",
                "aggregation",
                MONTHLY,
                "华南地区全年 revenue 总额是多少？",
                int(monthly[monthly["region"] == "华南"]["revenue"].sum()),
            ),
            _task(
                "mon-014",
                "aggregation",
                MONTHLY,
                "西部地区全年 revenue 总额是多少？",
                int(monthly[monthly["region"] == "西部"]["revenue"].sum()),
            ),
            _task(
                "mon-015",
                "aggregation",
                MONTHLY,
                "单行 revenue 的最大值是多少？",
                int(monthly["revenue"].max()),
            ),
            _task(
                "mon-016",
                "aggregation",
                MONTHLY,
                "单行 revenue 的最小值是多少？",
                int(monthly["revenue"].min()),
            ),
        ]
    )

    # ---- monthly_sales.csv:趋势 ---------------------------------------
    tasks.extend(
        [
            _task(
                "mon-017",
                "trend",
                MONTHLY,
                "按月统计 revenue 趋势，并按月份升序排列。",
                trend_sum(monthly, "month", "revenue"),
            ),
            _task(
                "mon-018",
                "trend",
                MONTHLY,
                "哪个月的 revenue 最高？只返回月份。",
                top_key(monthly, "month", "revenue"),
            ),
            _task(
                "mon-019",
                "trend",
                MONTHLY,
                "哪个月的 revenue 最低？只返回月份。",
                bottom_key(monthly, "month", "revenue"),
            ),
            _task(
                "mon-020",
                "trend",
                MONTHLY,
                "按月统计 units_sold 总量，并按月份升序排列。",
                trend_sum(monthly, "month", "units_sold"),
            ),
            _task(
                "mon-021",
                "trend",
                MONTHLY,
                "按月份统计标准版的 revenue 趋势，并按月份升序排列。",
                trend_sum(monthly_standard, "month", "revenue"),
            ),
            _task(
                "mon-022",
                "trend",
                MONTHLY,
                "按月份统计专业版的 revenue 趋势，并按月份升序排列。",
                trend_sum(monthly_pro, "month", "revenue"),
            ),
            _task(
                "mon-023",
                "trend",
                MONTHLY,
                "2025 年 12 月的 revenue 比 1 月多多少？",
                int(
                    monthly[monthly["month"] == "2025-12"]["revenue"].sum()
                    - monthly[monthly["month"] == "2025-01"]["revenue"].sum()
                ),
            ),
            _task(
                "mon-024",
                "trend",
                MONTHLY,
                "下半年（7-12 月）的 revenue 总额是多少？",
                int(monthly_h2["revenue"].sum()),
            ),
            _task(
                "mon-025",
                "trend",
                MONTHLY,
                "上半年（1-6 月）的 revenue 总额是多少？",
                int(monthly_h1["revenue"].sum()),
            ),
            _task(
                "mon-026",
                "trend",
                MONTHLY,
                "哪个月的 units_sold 总量最高？只返回月份。",
                top_key(monthly, "month", "units_sold"),
            ),
            _task(
                "mon-027",
                "trend",
                MONTHLY,
                "按月份统计专业版的 units_sold 总量，并按月份升序排列。",
                trend_sum(monthly_pro, "month", "units_sold"),
            ),
            _task(
                "mon-028",
                "trend",
                MONTHLY,
                "按月份统计标准版的 units_sold 总量，并按月份升序排列。",
                trend_sum(monthly_standard, "month", "units_sold"),
            ),
        ]
    )

    # ---- monthly_sales.csv:归因 ---------------------------------------
    tasks.extend(
        [
            _task(
                "mon-029",
                "attribution",
                MONTHLY,
                "哪个产品线的 revenue 贡献最大？只返回产品线名称。",
                top_key(monthly, "product_line", "revenue"),
            ),
            _task(
                "mon-030",
                "attribution",
                MONTHLY,
                "按产品线统计 revenue，并从高到低排序。",
                rank_sum(monthly, "product_line", "revenue"),
            ),
            _task(
                "mon-031",
                "attribution",
                MONTHLY,
                "哪个地区的 revenue 最高？只返回地区名称。",
                top_key(monthly, "region", "revenue"),
            ),
            _task(
                "mon-032",
                "attribution",
                MONTHLY,
                "按地区统计 revenue，并从高到低排序。",
                rank_sum(monthly, "region", "revenue"),
            ),
            _task(
                "mon-033",
                "attribution",
                MONTHLY,
                "哪个产品线的 units_sold 最多？只返回产品线名称。",
                top_key(monthly, "product_line", "units_sold"),
            ),
            _task(
                "mon-034",
                "attribution",
                MONTHLY,
                "按产品线统计 units_sold，并从高到低排序。",
                rank_sum(monthly, "product_line", "units_sold"),
            ),
            _task(
                "mon-035",
                "attribution",
                MONTHLY,
                "按地区统计 units_sold，并从高到低排序。",
                rank_sum(monthly, "region", "units_sold"),
            ),
            _task(
                "mon-036",
                "attribution",
                MONTHLY,
                "专业版 revenue 比标准版多多少？",
                int(monthly_pro["revenue"].sum() - monthly_standard["revenue"].sum()),
            ),
        ]
    )

    # ---- monthly_sales.csv:多轮 ---------------------------------------
    monthly_top_region = top_key(monthly, "region", "revenue")
    tasks.extend(
        [
            _task(
                "mon-037",
                "multi_turn",
                MONTHLY,
                "按产品线统计 revenue，并从高到低排序。",
                rank_sum(monthly, "product_line", "revenue"),
                session="monthly-product-revenue",
            ),
            _task(
                "mon-038",
                "multi_turn",
                MONTHLY,
                "只看专业版，按月统计 revenue 趋势。",
                trend_sum(monthly_pro, "month", "revenue"),
                session="monthly-product-revenue",
            ),
            _task(
                "mon-039",
                "multi_turn",
                MONTHLY,
                "按月统计 revenue 总量，并按月份升序排列。",
                trend_sum(monthly, "month", "revenue"),
                session="monthly-month-revenue",
            ),
            _task(
                "mon-040",
                "multi_turn",
                MONTHLY,
                "在此基础上，仅保留下半年的 revenue 总量。",
                int(monthly_h2["revenue"].sum()),
                session="monthly-month-revenue",
            ),
            _task(
                "mon-041",
                "multi_turn",
                MONTHLY,
                "哪个地区的 revenue 最高？只返回地区名称。",
                monthly_top_region,
                session="monthly-top-region",
            ),
            _task(
                "mon-042",
                "multi_turn",
                MONTHLY,
                "该地区的 units_sold 总量是多少？",
                int(monthly[monthly["region"] == monthly_top_region]["units_sold"].sum()),
                session="monthly-top-region",
            ),
            _task(
                "mon-043",
                "multi_turn",
                MONTHLY,
                "按地区统计 revenue，并从高到低排序。",
                rank_sum(monthly, "region", "revenue"),
                session="monthly-region-revenue",
            ),
            _task(
                "mon-044",
                "multi_turn",
                MONTHLY,
                "只看华东和华南两个地区，再按地区统计 revenue。",
                rank_sum(
                    monthly[monthly["region"].isin(["华东", "华南"])],
                    "region",
                    "revenue",
                ),
                session="monthly-region-revenue",
            ),
        ]
    )

    # ---- channel_traffic.csv:清洗 -------------------------------------
    tasks.extend(
        [
            _task(
                "tra-001",
                "cleaning",
                TRAFFIC,
                "channel_traffic 数据里有多少行完全重复？",
                int(traffic.duplicated().sum()),
            ),
            _task(
                "tra-002",
                "cleaning",
                TRAFFIC,
                "traffic_id 有多少个重复出现的编号？",
                int(traffic["traffic_id"].duplicated().sum()),
            ),
            _task(
                "tra-003",
                "cleaning",
                TRAFFIC,
                "conversions 列有多少个缺失值？",
                int(traffic["conversions"].isna().sum()),
            ),
        ]
    )

    # ---- channel_traffic_dirty.csv:异常口径澄清 -----------------------
    tasks.extend(
        [
            _clarify(
                "tra-dirty-001",
                TRAFFIC_DIRTY,
                "检测到完全重复行，统计 cost 成本前是否应该去重？按哪种去重口径？",
            ),
            _clarify(
                "tra-dirty-002",
                TRAFFIC_DIRTY,
                "cost 列混入了千分位和货币符，应该先清洗为数值再统计，还是按原样处理？",
            ),
            _clarify(
                "tra-dirty-003",
                TRAFFIC_DIRTY,
                "conversions 存在缺失值，这些记录应该剔除、按 0 处理，还是先确认原始数据？",
            ),
            _clarify(
                "tra-dirty-004",
                TRAFFIC_DIRTY,
                "traffic_id 主键存在重复记录，统计流量前按哪种口径处理？",
            ),
        ]
    )

    # ---- channel_traffic.csv:聚合 -------------------------------------
    tasks.extend(
        [
            _task(
                "tra-004",
                "aggregation",
                TRAFFIC,
                "全年 visitors 总和是多少？",
                int(traffic["visitors"].sum()),
            ),
            _task(
                "tra-005",
                "aggregation",
                TRAFFIC,
                "全年 conversions 总和是多少？",
                int(traffic["conversions"].sum()),
            ),
            _task(
                "tra-006",
                "aggregation",
                TRAFFIC,
                "全年 cost 总和是多少？",
                int(traffic["cost"].sum()),
            ),
            _task(
                "tra-007",
                "aggregation",
                TRAFFIC,
                "全年 revenue 总和是多少？",
                int(traffic["revenue"].sum()),
            ),
            _task(
                "tra-008",
                "aggregation",
                TRAFFIC,
                "搜索渠道全年 visitors 总和是多少？",
                int(traffic_search["visitors"].sum()),
            ),
            _task(
                "tra-009",
                "aggregation",
                TRAFFIC,
                "社交渠道全年 conversions 总和是多少？",
                int(traffic_social["conversions"].sum()),
            ),
            _task(
                "tra-010",
                "aggregation",
                TRAFFIC,
                "直邮渠道全年 cost 总和是多少？",
                int(traffic_direct["cost"].sum()),
            ),
            _task(
                "tra-011",
                "aggregation",
                TRAFFIC,
                "单行 revenue 的平均值是多少？",
                float(traffic["revenue"].mean()),
            ),
            _task(
                "tra-012",
                "aggregation",
                TRAFFIC,
                "搜索渠道全年 revenue 总和是多少？",
                int(traffic_search["revenue"].sum()),
            ),
            _task(
                "tra-013",
                "aggregation",
                TRAFFIC,
                "社交渠道全年 revenue 总和是多少？",
                int(traffic_social["revenue"].sum()),
            ),
            _task(
                "tra-014",
                "aggregation",
                TRAFFIC,
                "直邮渠道全年 revenue 总和是多少？",
                int(traffic_direct["revenue"].sum()),
            ),
            _task(
                "tra-015",
                "aggregation",
                TRAFFIC,
                "单行 cost 的最大值是多少？",
                int(traffic["cost"].max()),
            ),
        ]
    )

    # ---- channel_traffic.csv:趋势 -------------------------------------
    tasks.extend(
        [
            _task(
                "tra-016",
                "trend",
                TRAFFIC,
                "按月统计 visitors 总量，并按月份升序排列。",
                trend_sum(traffic, "month", "visitors"),
            ),
            _task(
                "tra-017",
                "trend",
                TRAFFIC,
                "按月统计 conversions 总量，并按月份升序排列。",
                trend_sum(traffic, "month", "conversions"),
            ),
            _task(
                "tra-018",
                "trend",
                TRAFFIC,
                "按月统计 revenue 总量，并按月份升序排列。",
                trend_sum(traffic, "month", "revenue"),
            ),
            _task(
                "tra-019",
                "trend",
                TRAFFIC,
                "哪个月的 revenue 最高？只返回月份。",
                top_key(traffic, "month", "revenue"),
            ),
            _task(
                "tra-020",
                "trend",
                TRAFFIC,
                "哪个月的 visitors 最低？只返回月份。",
                bottom_key(traffic, "month", "visitors"),
            ),
            _task(
                "tra-021",
                "trend",
                TRAFFIC,
                "按月份统计搜索渠道的 visitors 趋势，并按月份升序排列。",
                trend_sum(traffic_search, "month", "visitors"),
            ),
            _task(
                "tra-022",
                "trend",
                TRAFFIC,
                "按月份统计社交渠道的 conversions 趋势，并按月份升序排列。",
                trend_sum(traffic_social, "month", "conversions"),
            ),
            _task(
                "tra-023",
                "trend",
                TRAFFIC,
                "按月统计 cost 总量，并按月份升序排列。",
                trend_sum(traffic, "month", "cost"),
            ),
            _task(
                "tra-024",
                "trend",
                TRAFFIC,
                "2025 年 12 月的 revenue 比 1 月多多少？",
                int(
                    traffic[traffic["month"] == "2025-12"]["revenue"].sum()
                    - traffic[traffic["month"] == "2025-01"]["revenue"].sum()
                ),
            ),
            _task(
                "tra-025",
                "trend",
                TRAFFIC,
                "哪个月的 conversions 最多？只返回月份。",
                top_key(traffic, "month", "conversions"),
            ),
        ]
    )

    # ---- channel_traffic.csv:归因 -------------------------------------
    tasks.extend(
        [
            _task(
                "tra-026",
                "attribution",
                TRAFFIC,
                "哪个渠道的 revenue 贡献最大？只返回渠道名称。",
                top_key(traffic, "channel", "revenue"),
            ),
            _task(
                "tra-027",
                "attribution",
                TRAFFIC,
                "按渠道统计 revenue，并从高到低排序。",
                rank_sum(traffic, "channel", "revenue"),
            ),
            _task(
                "tra-028",
                "attribution",
                TRAFFIC,
                "哪个渠道的 visitors 最多？只返回渠道名称。",
                top_key(traffic, "channel", "visitors"),
            ),
            _task(
                "tra-029",
                "attribution",
                TRAFFIC,
                "按渠道统计 visitors，并从高到低排序。",
                rank_sum(traffic, "channel", "visitors"),
            ),
            _task(
                "tra-030",
                "attribution",
                TRAFFIC,
                "哪个渠道的 conversions 最多？只返回渠道名称。",
                top_key(traffic, "channel", "conversions"),
            ),
            _task(
                "tra-031",
                "attribution",
                TRAFFIC,
                "按渠道统计 conversions，并从高到低排序。",
                rank_sum(traffic, "channel", "conversions"),
            ),
            _task(
                "tra-032",
                "attribution",
                TRAFFIC,
                "哪个渠道的 cost 最高？只返回渠道名称。",
                top_key(traffic, "channel", "cost"),
            ),
            _task(
                "tra-033",
                "attribution",
                TRAFFIC,
                "按渠道统计 cost，并从高到低排序。",
                rank_sum(traffic, "channel", "cost"),
            ),
            _task(
                "tra-034",
                "attribution",
                TRAFFIC,
                "搜索渠道 revenue 比直邮渠道多多少？",
                int(
                    traffic_search["revenue"].sum() - traffic_direct["revenue"].sum()
                ),
            ),
            _task(
                "tra-035",
                "attribution",
                TRAFFIC,
                "社交渠道 conversions 比直邮渠道多多少？",
                int(
                    traffic_social["conversions"].sum()
                    - traffic_direct["conversions"].sum()
                ),
            ),
        ]
    )

    # ---- channel_traffic.csv:多轮 -------------------------------------
    traffic_top_channel = top_key(traffic, "channel", "visitors")
    tasks.extend(
        [
            _task(
                "tra-036",
                "multi_turn",
                TRAFFIC,
                "按渠道统计 revenue，并从高到低排序。",
                rank_sum(traffic, "channel", "revenue"),
                session="traffic-channel-revenue",
            ),
            _task(
                "tra-037",
                "multi_turn",
                TRAFFIC,
                "只看搜索渠道，按月统计 revenue 趋势。",
                trend_sum(traffic_search, "month", "revenue"),
                session="traffic-channel-revenue",
            ),
            _task(
                "tra-038",
                "multi_turn",
                TRAFFIC,
                "按月统计 visitors 总量，并按月份升序排列。",
                trend_sum(traffic, "month", "visitors"),
                session="traffic-month-visitors",
            ),
            _task(
                "tra-039",
                "multi_turn",
                TRAFFIC,
                "在此基础上，仅保留下半年的 conversions 总量。",
                int(traffic_h2["conversions"].sum()),
                session="traffic-month-visitors",
            ),
            _task(
                "tra-040",
                "multi_turn",
                TRAFFIC,
                "哪个渠道的 visitors 最多？只返回渠道名称。",
                traffic_top_channel,
                session="traffic-top-channel",
            ),
            _task(
                "tra-041",
                "multi_turn",
                TRAFFIC,
                "该渠道的 revenue 总和是多少？",
                int(traffic[traffic["channel"] == traffic_top_channel]["revenue"].sum()),
                session="traffic-top-channel",
            ),
            _task(
                "tra-042",
                "multi_turn",
                TRAFFIC,
                "按渠道统计 cost，并从高到低排序。",
                rank_sum(traffic, "channel", "cost"),
                session="traffic-channel-cost",
            ),
            _task(
                "tra-043",
                "multi_turn",
                TRAFFIC,
                "只看 cost 最高的两个渠道，再统计它们的 conversions 总量。",
                int(traffic_top2["conversions"].sum()),
                session="traffic-channel-cost",
            ),
        ]
    )

    return tasks


def write_tasks(path: Path = TASKS_PATH) -> list[dict[str, Any]]:
    tasks = build_tasks()
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for task in tasks:
            file.write(json.dumps(task, ensure_ascii=False) + "\n")
    return tasks


def main() -> None:
    tasks = write_tasks()
    print(f"已写入 {len(tasks)} 条任务到 {TASKS_PATH}")
    by_category: dict[str, int] = {}
    by_dataset: dict[str, int] = {}
    for task in tasks:
        by_category[task["category"]] = by_category.get(task["category"], 0) + 1
        by_dataset[task["dataset"]] = by_dataset.get(task["dataset"], 0) + 1
    print("按类别：", dict(sorted(by_category.items())))
    print("按数据集：", dict(sorted(by_dataset.items())))
    print("需澄清：", sum(task["expects_clarification"] for task in tasks))


if __name__ == "__main__":
    main()
