"""数据源连接器契约：读取文件为 DataFrame，再注册到 SQL 引擎。

新增数据源只需继承 :class:`BaseConnector` 并实现 :meth:`BaseConnector._read`
以及视图名约定；默认视图名由文件名清洗得到。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from prizm.engine.duckdb_engine import DuckDBEngine

_IDENTIFIER_RE = re.compile(r"[^0-9A-Za-z_]+")


def sanitize_view_name(raw: str) -> str:
    """把任意名称清洗成 DuckDB 可用标识符。"""
    name = _IDENTIFIER_RE.sub("_", raw.strip()).strip("_").lower()
    if not name:
        return "dataset"
    if name[0].isdigit():
        name = f"t_{name}"
    return name


class BaseConnector(ABC):
    """把数据文件读成 DataFrame，并注册为 DuckDB 视图。"""

    def __init__(self, path: str | Path, view_name: str | None = None) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"数据文件不存在: {self.path}")
        self._view_name = sanitize_view_name(view_name or self.path.stem)
        self._frame: pd.DataFrame | None = None

    @property
    def view_name(self) -> str:
        return self._view_name

    def load(self) -> pd.DataFrame:
        """读取并缓存 DataFrame，保证会话内只加载一次。"""
        if self._frame is None:
            self._frame = self._read()
        return self._frame

    def register(self, engine: DuckDBEngine) -> None:
        engine.register_dataframe(self.view_name, self.load())

    @abstractmethod
    def _read(self) -> pd.DataFrame:
        """真正读取文件的实现。"""
