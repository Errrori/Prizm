"""CSV 连接器：按 UTF-8(BOM) → GBK 的顺序尝试解码。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from prizm.datasource.base import BaseConnector

_CANDIDATE_ENCODINGS = ("utf-8-sig", "gbk")


class CsvConnector(BaseConnector):
    def __init__(
        self,
        path: str | Path,
        view_name: str | None = None,
        *,
        encoding: str | None = None,
        **read_csv_kwargs: Any,
    ) -> None:
        super().__init__(path, view_name)
        self._encoding = encoding
        self._read_kwargs = read_csv_kwargs

    def _read(self) -> pd.DataFrame:
        # DEBT-003: 编码采用启发式回退, 未做 charset 检测。
        if self._encoding is not None:
            return pd.read_csv(self.path, encoding=self._encoding, **self._read_kwargs)

        last_error: UnicodeDecodeError | None = None
        for encoding in _CANDIDATE_ENCODINGS:
            try:
                return pd.read_csv(self.path, encoding=encoding, **self._read_kwargs)
            except UnicodeDecodeError as exc:
                last_error = exc

        tried = ", ".join(_CANDIDATE_ENCODINGS)
        raise ValueError(f"无法解码 CSV 文件 {self.path}（已尝试编码: {tried}）") from last_error
