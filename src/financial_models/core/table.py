"""Lightweight table structure with optional pandas integration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Union

try:  # pragma: no cover - executed when pandas is available
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover - pandas may not be installed in tests
    _pd = None  # type: ignore


Number = Union[float, int]


def _ensure_length(name: str, values: Iterable[Number], expected: int) -> List[float]:
    data = [float(v) for v in values]
    if len(data) != expected:
        raise ValueError(f"Column '{name}' expected {expected} values, received {len(data)}")
    return data


@dataclass
class Table:
    """Container for tabular numeric data."""

    index: List[Any]
    data: MutableMapping[str, List[float]]
    index_name: str = "Year"

    def __post_init__(self) -> None:
        length = len(self.index)
        for name, values in list(self.data.items()):
            self.data[name] = _ensure_length(name, values, length)

    def column(self, name: str) -> List[float]:
        return list(self.data[name])

    def columns(self) -> List[str]:
        return list(self.data.keys())

    def as_dict(self) -> Dict[str, List[float]]:
        return {key: list(values) for key, values in self.data.items()}

    def to_frame(self):  # type: ignore[override]
        if _pd is None:
            raise RuntimeError("pandas is not installed; install it to obtain a DataFrame")
        frame = _pd.DataFrame(self.data, index=self.index)  # type: ignore[attr-defined]
        frame.index.name = self.index_name
        return frame

    def to_csv(self, path: Path) -> None:
        if _pd is not None:
            self.to_frame().to_csv(path)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [self.index_name] + self.columns()
        rows = []
        for position, idx in enumerate(self.index):
            row = {self.index_name: idx}
            for column in self.columns():
                row[column] = self.data[column][position]
            rows.append(row)
        import csv

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def with_columns(self, **columns: Iterable[Number]) -> "Table":
        updated: Dict[str, List[float]] = self.as_dict()
        for name, values in columns.items():
            updated[name] = _ensure_length(name, values, len(self.index))
        return Table(list(self.index), updated, self.index_name)

    def select(self, names: Iterable[str]) -> "Table":
        selected = {name: self.data[name] for name in names}
        return Table(list(self.index), {k: list(v) for k, v in selected.items()}, self.index_name)

    def rounded(
        self,
        decimals: int,
        *,
        exclude_keywords: Iterable[str] = (),
    ) -> "Table":
        exclude = tuple(exclude_keywords)

        def _should_round(name: Any) -> bool:
            if not isinstance(name, str):
                return True
            return not any(keyword in name for keyword in exclude)

        rounded_data: Dict[str, List[float]] = {}
        for name, values in self.data.items():
            if _should_round(name):
                rounded_data[name] = [round(float(value), decimals) for value in values]
            else:
                rounded_data[name] = list(values)
        return Table(list(self.index), rounded_data, self.index_name)


def build_table(index: Iterable[Any], columns: Mapping[str, Iterable[Number]], index_name: str = "Year") -> Table:
    return Table(list(index), {k: list(map(float, v)) for k, v in columns.items()}, index_name)


__all__ = ["Table", "build_table"]
