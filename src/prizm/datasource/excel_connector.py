"""Excel 连接器：用 openpyxl 读取单个 .xlsx 工作表（暂不支持旧版 .xls）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from prizm.datasource.base import BaseConnector

_SUPPORTED_SUFFIX = ".xlsx"


def _ensure_xlsx(path: Path) -> None:
    # DEBT-021: 仅支持 .xlsx 单 sheet, 未处理 .xls、合并单元格与多级表头。
    if path.suffix.lower() != _SUPPORTED_SUFFIX:
        raise ValueError(
            f"仅支持 .xlsx 格式（{path.name}）；旧版 .xls 请另存为 .xlsx 后重试"
        )
    if not path.is_file():
        raise FileNotFoundError(f"数据文件不存在: {path}")


def list_excel_sheets(path: str | Path) -> tuple[str, ...]:
    """列出工作簿中的工作表名称，供 UI 选择。"""
    file = Path(path)
    _ensure_xlsx(file)
    with pd.ExcelFile(file, engine="openpyxl") as workbook:
        return tuple(str(name) for name in workbook.sheet_names)


class ExcelConnector(BaseConnector):
    def __init__(
        self,
        path: str | Path,
        view_name: str | None = None,
        *,
        sheet_name: str | None = None,
        **read_excel_kwargs: Any,
    ) -> None:
        super().__init__(path, view_name)
        _ensure_xlsx(self.path)
        self._sheet_name = sheet_name
        self._read_kwargs = read_excel_kwargs

    @property
    def sheet_name(self) -> str | None:
        return self._sheet_name

    def _read(self) -> pd.DataFrame:
        sheets = list_excel_sheets(self.path)
        if not sheets:
            raise ValueError(f"Excel 文件没有工作表: {self.path.name}")
        sheet = self._sheet_name or sheets[0]
        if sheet not in sheets:
            available = "、".join(sheets)
            raise ValueError(
                f"工作表不存在: {sheet}（{self.path.name} 可用工作表: {available}）"
            )
        kwargs = dict(self._read_kwargs)
        kwargs.setdefault("engine", "openpyxl")
        return pd.read_excel(self.path, sheet_name=sheet, **kwargs)
