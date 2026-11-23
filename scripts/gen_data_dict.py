"""从 CSV 生成数据字典 JSONL，供人工校对或上传到 RAG。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from prizm.datasource.csv_connector import CsvConnector
from prizm.profiling.profiler import profile_frame
from prizm.rag.ingest import build_data_dictionary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 CSV 数据字典")
    parser.add_argument("csv", type=Path, help="CSV 文件路径")
    parser.add_argument("--source", default=None, help="字典中展示的数据集名称")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 JSONL 路径；省略则打印到 stdout",
    )
    args = parser.parse_args(argv)

    try:
        connector = CsvConnector(args.csv)
        frame = connector.load()
        entries = build_data_dictionary(
            frame,
            profile=profile_frame(frame),
            source=args.source or args.csv.name,
        )
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    lines = [json.dumps(asdict(entry), ensure_ascii=False) for entry in entries]
    if args.out is None:
        print("\n".join(lines))
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"已写入 {len(entries)} 条数据字典: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
