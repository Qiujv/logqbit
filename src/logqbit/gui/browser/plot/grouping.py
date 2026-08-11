"""Stable dataframe grouping for stored-record plots."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PlotGroup:
    """One dataframe slice and its user-facing group label."""

    label: str
    frame: pd.DataFrame


def iter_plot_groups(
    frame: pd.DataFrame,
    groupby: Sequence[str],
) -> Iterator[PlotGroup]:
    """Yield plot groups in first-observed order, retaining missing values."""
    columns = tuple(groupby)
    if not columns:
        yield PlotGroup("", frame)
        return

    grouper: str | list[str] = columns[0] if len(columns) == 1 else list(columns)
    grouped = frame.groupby(grouper, sort=False, dropna=False, observed=True)
    for key, group_frame in grouped:
        values = (key,) if len(columns) == 1 else tuple(key)
        label = ", ".join(
            f"{column}={_format_group_value(value)}"
            for column, value in zip(columns, values, strict=True)
        )
        yield PlotGroup(label, group_frame)


def _format_group_value(value: object) -> str:
    try:
        if bool(pd.isna(value)):
            return "<NA>"
    except (TypeError, ValueError):
        pass
    return str(value)
