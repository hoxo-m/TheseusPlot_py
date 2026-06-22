"""ShipOfTheseus public object."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.api.types import is_numeric_dtype, is_string_dtype

from theseusplot._config import ContinuousConfig, continuous_config

_OUTCOME_COLUMN = ".outcome"
_MISSING_LABEL = "(Missing)"
_MIN_BREAK_COUNT = 2
_REFITTED_COLOR = "#00BFC4"
_ORIGINAL_SIZE_COLOR = "#7CAE00"
_REFITTED_SIZE_COLOR = "#C77CFF"
_POSITIVE_COLOR = "#00BFC4"
_NEGATIVE_COLOR = "#F8766D"


class ShipOfTheseus:
    """Container for data and methods used to create Theseus plots."""

    def __init__(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        outcome: str,
        labels: Sequence[str],
        y_label: str | None,
        digits: int,
        text_size: float,
        x_label: str | None = None,
    ) -> None:
        self._validate_inputs(data1, data2, outcome, labels)

        self._data1 = self._prepare_input_data(data1, outcome)
        self._data2 = self._prepare_input_data(data2, outcome)
        self.outcome = outcome
        self.labels = (labels[0], labels[1])
        self.x_label = x_label
        self.y_label = y_label
        self.digits = digits
        self.text_size = text_size
        self._cache: dict[tuple[Any, ...], Any] = {}

    @staticmethod
    def _validate_inputs(
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        outcome: str,
        labels: Sequence[str],
    ) -> None:
        if not isinstance(data1, pd.DataFrame):
            msg = "data1 must be a pandas DataFrame."
            raise TypeError(msg)
        if not isinstance(data2, pd.DataFrame):
            msg = "data2 must be a pandas DataFrame."
            raise TypeError(msg)
        if outcome not in data1.columns:
            msg = f"outcome column {outcome!r} is missing from data1."
            raise ValueError(msg)
        if outcome not in data2.columns:
            msg = f"outcome column {outcome!r} is missing from data2."
            raise ValueError(msg)
        if len(labels) != 2:
            msg = "labels must contain exactly two values."
            raise ValueError(msg)
        if data1[outcome].isna().any() or data2[outcome].isna().any():
            msg = "outcome values must not contain missing values."
            raise ValueError(msg)

    @staticmethod
    def _prepare_input_data(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
        prepared = data.copy(deep=True)
        for column in prepared.columns:
            if column == outcome:
                continue
            prepared[column] = ShipOfTheseus._fill_missing_categories(
                prepared[column],
            )
        prepared[_OUTCOME_COLUMN] = prepared[outcome]
        return prepared

    @staticmethod
    def _fill_missing_categories(series: pd.Series) -> pd.Series:
        if isinstance(series.dtype, pd.CategoricalDtype):
            result = series.copy()
            if _MISSING_LABEL not in result.cat.categories:
                result = result.cat.add_categories([_MISSING_LABEL])
            return result.fillna(_MISSING_LABEL)
        if is_string_dtype(series.dtype) or series.dtype == object:
            return series.fillna(_MISSING_LABEL)
        return series

    def table(
        self,
        column: str,
        n: int | float = float("inf"),
        continuous: ContinuousConfig | None = None,
    ) -> pd.DataFrame:
        """Generate a contribution table for a column."""

        self._validate_column(column)
        limit = self._normalize_n(n)
        continuous_config_value = continuous or continuous_config()

        data_contrib = self._compute_contribution(column, continuous_config_value)
        data_info = self._compute_info(column, continuous_config_value)
        result = data_contrib.merge(data_info, on="items", how="left")

        is_factor = isinstance(result["items"].dtype, pd.CategoricalDtype)
        if is_factor:
            result = result.sort_values("items", kind="stable").reset_index(drop=True)
            levels = list(result["items"].cat.categories)
        else:
            result = self._sort_by_abs_contrib(result)
            levels = []

        n_items = len(result)
        if n_items > limit:
            other_count = n_items - limit + 1
            sorted_result = self._sort_by_abs_contrib(result)
            result_head = sorted_result.head(limit - 1).copy()
            result_head["items"] = result_head["items"].astype(str)

            result_tail = sorted_result.tail(other_count).copy()
            other_label = f"Sum of {other_count} other attributes"
            result_other = pd.DataFrame(
                [
                    {
                        "items": other_label,
                        "contrib": result_tail["contrib"].sum(),
                        "n1": int(result_tail["n1"].sum()),
                        "n2": int(result_tail["n2"].sum()),
                        "x1": result_tail["x1"].sum(),
                        "x2": result_tail["x2"].sum(),
                    },
                ],
            )
            result_other["rate1"] = self._safe_rate(
                result_other.loc[0, "x1"],
                result_other.loc[0, "n1"],
            )
            result_other["rate2"] = self._safe_rate(
                result_other.loc[0, "x2"],
                result_other.loc[0, "n2"],
            )
            result = pd.concat([result_head, result_other], ignore_index=True)
            if is_factor:
                result["items"] = pd.Categorical(
                    result["items"],
                    categories=[*levels, other_label],
                    ordered=True,
                )
                result = result.sort_values("items", kind="stable").reset_index(
                    drop=True,
                )

        result = result.rename(columns={"items": column})
        return result[
            [column, "contrib", "n1", "n2", "x1", "x2", "rate1", "rate2"]
        ]

    def plot(
        self,
        column: str,
        n: int = 10,
        main_item: str | None = None,
        bar_max_value: float | None = None,
        levels: Sequence[str] | None = None,
        continuous: ContinuousConfig | None = None,
        ax: Any | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """Generate a Theseus plot for a column."""

        continuous_config_value = continuous or continuous_config()
        score1, score2 = self._compute_scores(column)
        table = self.table(column, n=n, continuous=continuous_config_value)
        plot_data = self._plot_contribution_data(
            table=table,
            column=column,
            levels=levels,
        )
        size_data = self._plot_size_data(
            table=plot_data,
            column=column,
            main_item=main_item,
            bar_max_value=bar_max_value,
            max_score=max(score1, score2),
        )
        waterfall = self._waterfall_data(
            items=plot_data[column].astype(str).tolist(),
            contributions=plot_data["contrib"].astype(float).tolist(),
            start=score1,
        )
        return self._draw_plot(
            waterfall=waterfall,
            size_data=size_data,
            column=column,
            ax=ax,
            figsize=figsize,
        )

    def plot_flip(
        self,
        column: str,
        n: int = 10,
        main_item: str | None = None,
        bar_max_value: float | None = None,
        levels: Sequence[str] | None = None,
        continuous: ContinuousConfig | None = None,
        ax: Any | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """Generate a horizontally oriented Theseus plot for a column."""

        continuous_config_value = continuous or continuous_config()
        score1, score2 = self._compute_scores(column)
        table = self.table(column, n=n, continuous=continuous_config_value)
        table = table.copy()
        table["contrib"] = -table["contrib"]
        plot_data = self._plot_flip_contribution_data(
            table=table,
            column=column,
            levels=levels,
        )
        size_data = self._plot_size_data(
            table=plot_data,
            column=column,
            main_item=main_item,
            bar_max_value=bar_max_value,
            max_score=max(score1, score2),
        )
        waterfall = self._waterfall_data(
            items=plot_data[column].astype(str).tolist(),
            contributions=plot_data["contrib"].astype(float).tolist(),
            start=score2,
            start_label=self.labels[1],
            end_label=self.labels[0],
        )
        return self._draw_plot_flip(
            waterfall=waterfall,
            size_data=size_data,
            column=column,
            ax=ax,
            figsize=figsize,
        )

    def _compute_scores(self, column: str) -> tuple[float, float]:
        key = ("scores", column)
        if key not in self._cache:
            self._cache[key] = (
                float(self._data1[_OUTCOME_COLUMN].mean()),
                float(self._data2[_OUTCOME_COLUMN].mean()),
            )
        return cast(tuple[float, float], self._cache[key])

    def _to_factor(
        self,
        column: str,
        continuous: ContinuousConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        key = ("to_factor", column, continuous)
        if key not in self._cache:
            breaks = self._continuous_breaks(column, continuous)
            data1 = self._data1.copy()
            data2 = self._data2.copy()
            data1[column] = self._cut_to_categorical(data1[column], breaks)
            data2[column] = self._cut_to_categorical(data2[column], breaks)
            self._cache[key] = (data1, data2)
        data1_cached, data2_cached = cast(
            tuple[pd.DataFrame, pd.DataFrame],
            self._cache[key],
        )
        return data1_cached.copy(), data2_cached.copy()

    def _compute_contribution(
        self,
        column: str,
        continuous: ContinuousConfig,
    ) -> pd.DataFrame:
        key = ("contribution", column, continuous)
        if key in self._cache:
            cached = cast(pd.DataFrame, self._cache[key])
            return cached.copy()

        data1, data2 = self._data_for_column(column, continuous)
        grouped1 = self._summarize_by_column(data1, column)
        grouped2 = self._summarize_by_column(data2, column)

        score1, score2 = self._compute_scores(column)
        amounts = self._compute_replacement_amounts(
            original=grouped1,
            refitted=grouped2,
            original_score=score1,
            refitted_score=score2,
        )
        result = self._average_replacement_amounts(
            amounts=amounts,
            original_items=grouped1["items"],
            refitted_items=grouped2["items"],
        )
        result["contrib"] = self._scale_contributions(
            result["contrib"],
            overall_diff=score2 - score1,
        )

        self._cache[key] = result.copy()
        return result

    @classmethod
    def _compute_replacement_amounts(
        cls,
        original: pd.DataFrame,
        refitted: pd.DataFrame,
        original_score: float,
        refitted_score: float,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for item in refitted["items"]:
            replaced = cls._replace_or_append_group(original, refitted, item)
            rows.append(
                {
                    "items": item,
                    "amount": cls._score_from_summary(replaced) - original_score,
                },
            )

        for item in original["items"]:
            replaced = cls._replace_or_append_group(refitted, original, item)
            rows.append(
                {
                    "items": item,
                    "amount": refitted_score - cls._score_from_summary(replaced),
                },
            )

        return pd.DataFrame(rows, columns=["items", "amount"])

    @classmethod
    def _average_replacement_amounts(
        cls,
        amounts: pd.DataFrame,
        original_items: pd.Series,
        refitted_items: pd.Series,
    ) -> pd.DataFrame:
        amounts = amounts.copy()
        dtype = cls._combined_categorical_dtype(original_items, refitted_items)
        if dtype is not None:
            amounts["items"] = pd.Categorical(
                amounts["items"],
                categories=dtype.categories,
                ordered=dtype.ordered,
            )

        return (
            amounts.groupby("items", observed=True, sort=False)["amount"]
            .mean()
            .reset_index()
            .rename(columns={"amount": "contrib"})
        )

    @staticmethod
    def _scale_contributions(
        contributions: pd.Series,
        overall_diff: float,
    ) -> pd.Series:
        raw_total = float(contributions.sum())
        if np.isclose(raw_total, 0.0):
            scaled = 0.0 if np.isclose(overall_diff, 0.0) else np.nan
            return pd.Series(scaled, index=contributions.index, dtype=float)
        return cast(pd.Series, overall_diff * contributions / raw_total)

    def _compute_info(self, column: str, continuous: ContinuousConfig) -> pd.DataFrame:
        key = ("info", column, continuous)
        if key in self._cache:
            cached = cast(pd.DataFrame, self._cache[key])
            return cached.copy()

        data1, data2 = self._data_for_column(column, continuous)
        data1_info = self._summarize_info(data1, column, suffix="1")
        data2_info = self._summarize_info(data2, column, suffix="2")
        result = data1_info.merge(data2_info, on="items", how="outer", sort=False)

        dtype = self._combined_categorical_dtype(
            data1_info["items"],
            data2_info["items"],
        )
        if dtype is not None:
            result["items"] = pd.Categorical(
                result["items"],
                categories=dtype.categories,
                ordered=dtype.ordered,
            )

        for column_name in ("n1", "n2"):
            result[column_name] = result[column_name].fillna(0).astype(int)
        for column_name in ("x1", "x2"):
            result[column_name] = result[column_name].fillna(0)

        result = result[["items", "n1", "n2", "x1", "x2", "rate1", "rate2"]]
        self._cache[key] = result.copy()
        return result

    def _compute_size(
        self,
        column: str,
        target: Sequence[str],
        continuous: ContinuousConfig,
    ) -> pd.DataFrame:
        data1, data2 = self._data_for_column(column, continuous)
        target_items = [str(item) for item in target]

        data1_size = self._count_target_items(
            data=data1,
            column=column,
            target=target_items,
            label=self.labels[0],
        )
        data2_size = self._count_target_items(
            data=data2,
            column=column,
            target=target_items,
            label=self.labels[1],
        )
        item_names = set(data1_size["items"].astype(str)).union(
            data2_size["items"].astype(str),
        )
        other_names = [item for item in target_items if item not in item_names]

        if not other_names:
            return pd.concat([data1_size, data2_size], ignore_index=True)

        rows = [data1_size, data2_size]
        for item in other_names:
            rows.append(
                self._count_other_items(
                    data=data1,
                    column=column,
                    target=target_items,
                    item=item,
                    label=self.labels[0],
                ),
            )
            rows.append(
                self._count_other_items(
                    data=data2,
                    column=column,
                    target=target_items,
                    item=item,
                    label=self.labels[1],
                ),
            )
        return pd.concat(rows, ignore_index=True)

    def _plot_contribution_data(
        self,
        table: pd.DataFrame,
        column: str,
        levels: Sequence[str] | None,
    ) -> pd.DataFrame:
        data = table[[column, "contrib", "n1", "n2"]].copy()
        is_factor = isinstance(data[column].dtype, pd.CategoricalDtype)
        if is_factor:
            data = data.sort_values(column, kind="stable")
        else:
            data = data.sort_values("contrib", kind="stable")

        if levels is not None:
            level_frame = pd.DataFrame({column: [str(level) for level in levels]})
            data[column] = data[column].astype(str)
            data = level_frame.merge(data, on=column, how="inner", sort=False)

        return data.reset_index(drop=True)

    def _plot_flip_contribution_data(
        self,
        table: pd.DataFrame,
        column: str,
        levels: Sequence[str] | None,
    ) -> pd.DataFrame:
        data = table[[column, "contrib", "n1", "n2"]].copy()
        is_factor = isinstance(data[column].dtype, pd.CategoricalDtype)
        if is_factor:
            data = data.sort_values(column, ascending=False, kind="stable")
        else:
            data = data.sort_values("contrib", kind="stable")

        if levels is not None:
            level_frame = pd.DataFrame(
                {column: [str(level) for level in reversed(levels)]},
            )
            data[column] = data[column].astype(str)
            data = level_frame.merge(data, on=column, how="inner", sort=False)

        return data.reset_index(drop=True)

    def _plot_size_data(
        self,
        table: pd.DataFrame,
        column: str,
        main_item: str | None,
        bar_max_value: float | None,
        max_score: float,
    ) -> pd.DataFrame:
        size_data = pd.concat(
            [
                pd.DataFrame(
                    {
                        "items": table[column].astype(str),
                        "n": table["n1"].astype(float),
                        "type": self.labels[0],
                    },
                ),
                pd.DataFrame(
                    {
                        "items": table[column].astype(str),
                        "n": table["n2"].astype(float),
                        "type": self.labels[1],
                    },
                ),
            ],
            ignore_index=True,
        )
        if table.empty:
            size_data["scaled_n"] = 0.0
            return size_data

        max_amount, n_max = self._size_scale_reference(
            table=table,
            column=column,
            main_item=main_item,
            bar_max_value=bar_max_value,
            max_score=max_score,
        )
        if np.isclose(n_max, 0.0) or np.isclose(max_amount, 0.0):
            size_data["scaled_n"] = 0.0
        else:
            size_data["scaled_n"] = size_data["n"] / n_max * max_amount
        return size_data

    def _size_scale_reference(
        self,
        table: pd.DataFrame,
        column: str,
        main_item: str | None,
        bar_max_value: float | None,
        max_score: float,
        bar_min_ratio: float = 0.15,
    ) -> tuple[float, float]:
        data = table.copy()
        data["_max_n"] = data[["n1", "n2"]].max(axis=1).astype(float)
        data["_abs_contrib"] = (data["contrib"].astype(float) * 100).abs()

        if main_item is not None:
            row = data[data[column].astype(str) == str(main_item)]
            if row.empty:
                msg = f"main_item {main_item!r} is not present in the plot data."
                raise ValueError(msg)
            return float(row["_abs_contrib"].iloc[0]), float(row["_max_n"].iloc[0])

        if bar_max_value is not None:
            return abs(float(bar_max_value)), float(data["_max_n"].max())

        return abs(float(max_score)) * 100 * bar_min_ratio, float(data["_max_n"].max())

    def _waterfall_data(
        self,
        items: Sequence[str],
        contributions: Sequence[float],
        start: float,
        start_label: str | None = None,
        end_label: str | None = None,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        cumulative = start * 100
        start_label = self.labels[0] if start_label is None else start_label
        end_label = self.labels[1] if end_label is None else end_label
        rows.append(
            {
                "items": start_label,
                "bottom": 0.0,
                "height": cumulative,
                "amount": cumulative,
                "cumulative": cumulative,
                "kind": "total",
            },
        )

        for item, contribution in zip(items, contributions, strict=True):
            amount = contribution * 100
            bottom = cumulative if amount >= 0 else cumulative + amount
            cumulative += amount
            rows.append(
                {
                    "items": item,
                    "bottom": bottom,
                    "height": abs(amount),
                    "amount": amount,
                    "cumulative": cumulative,
                    "kind": "contribution",
                },
            )

        rows.append(
            {
                "items": end_label,
                "bottom": 0.0,
                "height": cumulative,
                "amount": cumulative,
                "cumulative": cumulative,
                "kind": "total",
            },
        )
        return pd.DataFrame(rows)

    def _draw_plot(
        self,
        waterfall: pd.DataFrame,
        size_data: pd.DataFrame,
        column: str,
        ax: Any | None,
        figsize: tuple[float, float] | None,
    ) -> Any:
        plt = self._load_pyplot()
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or self._default_figsize(waterfall))
        else:
            fig = ax.figure

        positions = cast(
            NDArray[np.int64],
            np.arange(len(waterfall), dtype=np.int64),
        )
        item_to_position = {
            str(item): index for index, item in enumerate(waterfall["items"])
        }

        self._draw_size_bars(ax=ax, size_data=size_data, positions=item_to_position)
        self._draw_waterfall_bars(ax=ax, waterfall=waterfall, positions=positions)
        self._draw_connectors(ax=ax, waterfall=waterfall, positions=positions)

        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(waterfall["items"].astype(str), rotation=45, ha="right")
        ax.set_ylabel(self.y_label or "")
        ax.set_xlabel(self.x_label or "")
        ax.set_title("")
        ax.margins(x=0.02)
        fig.tight_layout()
        return fig, ax

    def _draw_plot_flip(
        self,
        waterfall: pd.DataFrame,
        size_data: pd.DataFrame,
        column: str,
        ax: Any | None,
        figsize: tuple[float, float] | None,
    ) -> Any:
        plt = self._load_pyplot()
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize or self._default_flip_figsize(waterfall),
            )
        else:
            fig = ax.figure

        positions = cast(
            NDArray[np.int64],
            np.arange(len(waterfall), dtype=np.int64),
        )
        item_to_position = {
            str(item): index for index, item in enumerate(waterfall["items"])
        }

        self._draw_size_bars_horizontal(
            ax=ax,
            size_data=size_data,
            positions=item_to_position,
        )
        self._draw_waterfall_bars_horizontal(
            ax=ax,
            waterfall=waterfall,
            positions=positions,
        )
        self._draw_connectors_horizontal(
            ax=ax,
            waterfall=waterfall,
            positions=positions,
        )

        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_yticks(positions)
        ax.set_yticklabels(waterfall["items"].astype(str))
        ax.set_xlabel(self.y_label or "")
        ax.set_ylabel(self.x_label or "")
        ax.set_title("")
        ax.margins(y=0.02)
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def _load_pyplot() -> Any:
        try:
            return cast(Any, importlib.import_module("matplotlib.pyplot"))
        except ModuleNotFoundError as exc:
            msg = "matplotlib is required to use plotting methods."
            raise ModuleNotFoundError(msg) from exc

    def _draw_size_bars(
        self,
        ax: Any,
        size_data: pd.DataFrame,
        positions: dict[str, int],
    ) -> None:
        width = 0.22
        offsets = {self.labels[0]: -width / 1.5, self.labels[1]: width / 1.5}
        colors = {
            self.labels[0]: _ORIGINAL_SIZE_COLOR,
            self.labels[1]: _REFITTED_SIZE_COLOR,
        }
        for _, row in size_data.iterrows():
            item = str(row["items"])
            if item not in positions:
                continue
            group = str(row["type"])
            ax.bar(
                positions[item] + offsets[group],
                row["scaled_n"],
                width=width,
                color=colors[group],
                linewidth=0,
                zorder=1,
            )

    def _draw_size_bars_horizontal(
        self,
        ax: Any,
        size_data: pd.DataFrame,
        positions: dict[str, int],
    ) -> None:
        height = 0.22
        offsets = {self.labels[0]: height / 1.5, self.labels[1]: -height / 1.5}
        colors = {
            self.labels[0]: _ORIGINAL_SIZE_COLOR,
            self.labels[1]: _REFITTED_SIZE_COLOR,
        }
        for _, row in size_data.iterrows():
            item = str(row["items"])
            if item not in positions:
                continue
            group = str(row["type"])
            ax.barh(
                positions[item] + offsets[group],
                row["scaled_n"],
                height=height,
                color=colors[group],
                linewidth=0,
                zorder=1,
            )

    def _draw_waterfall_bars(
        self,
        ax: Any,
        waterfall: pd.DataFrame,
        positions: NDArray[np.int64],
    ) -> None:
        colors = [
            _REFITTED_COLOR
            if row["kind"] == "total"
            else self._contribution_color(float(row["amount"]))
            for _, row in waterfall.iterrows()
        ]
        ax.bar(
            positions,
            waterfall["height"],
            bottom=waterfall["bottom"],
            width=0.62,
            color=colors,
            edgecolor="#333333",
            linewidth=0.6,
            zorder=3,
        )
        for position, (_, row) in zip(positions, waterfall.iterrows(), strict=True):
            value = self._format_plot_value(float(row["amount"]))
            y = float(row["bottom"]) + float(row["height"])
            va = "bottom"
            if float(row["amount"]) < 0:
                y = float(row["bottom"])
                va = "top"
            ax.text(
                position,
                y,
                value,
                ha="center",
                va=va,
                fontsize=9 * self.text_size,
                zorder=4,
            )

    def _draw_waterfall_bars_horizontal(
        self,
        ax: Any,
        waterfall: pd.DataFrame,
        positions: NDArray[np.int64],
    ) -> None:
        colors = [
            _REFITTED_COLOR
            if row["kind"] == "total"
            else self._contribution_color(-float(row["amount"]))
            for _, row in waterfall.iterrows()
        ]
        ax.barh(
            positions,
            waterfall["height"],
            left=waterfall["bottom"],
            height=0.62,
            color=colors,
            edgecolor="#333333",
            linewidth=0.6,
            zorder=3,
        )
        for position, (_, row) in zip(positions, waterfall.iterrows(), strict=True):
            amount = float(row["amount"])
            display_amount = -amount if row["kind"] == "contribution" else amount
            value = self._format_plot_value(display_amount)
            x = float(row["bottom"]) + float(row["height"])
            ha = "left"
            if amount < 0:
                x = float(row["bottom"])
                ha = "right"
            ax.text(
                x,
                position,
                value,
                ha=ha,
                va="center",
                fontsize=9 * self.text_size,
                zorder=4,
            )

    @staticmethod
    def _contribution_color(amount: float) -> str:
        return _NEGATIVE_COLOR if amount < 0 else _POSITIVE_COLOR

    @staticmethod
    def _draw_connectors(
        ax: Any,
        waterfall: pd.DataFrame,
        positions: NDArray[np.int64],
    ) -> None:
        for index in range(len(waterfall) - 2):
            y = float(waterfall.loc[index, "cumulative"])
            ax.plot(
                [positions[index] + 0.31, positions[index + 1] - 0.31],
                [y, y],
                color="#666666",
                linewidth=0.8,
                zorder=2,
            )

    @staticmethod
    def _draw_connectors_horizontal(
        ax: Any,
        waterfall: pd.DataFrame,
        positions: NDArray[np.int64],
    ) -> None:
        for index in range(len(waterfall) - 2):
            x = float(waterfall.loc[index, "cumulative"])
            ax.plot(
                [x, x],
                [positions[index] + 0.31, positions[index + 1] - 0.31],
                color="#666666",
                linewidth=0.8,
                zorder=2,
            )

    @staticmethod
    def _default_figsize(waterfall: pd.DataFrame) -> tuple[float, float]:
        return max(6.0, len(waterfall) * 0.75), 4.5

    @staticmethod
    def _default_flip_figsize(waterfall: pd.DataFrame) -> tuple[float, float]:
        return 6.5, max(4.5, len(waterfall) * 0.45)

    def _format_plot_value(self, value: float) -> str:
        rounded = round(value, self.digits)
        return f"{rounded:g}"

    @staticmethod
    def _count_target_items(
        data: pd.DataFrame,
        column: str,
        target: Sequence[str],
        label: str,
    ) -> pd.DataFrame:
        counts = (
            data[data[column].astype(str).isin(target)]
            .groupby(column, observed=True, sort=False)
            .size()
            .reset_index(name="n")
            .rename(columns={column: "items"})
        )
        counts["items"] = counts["items"].astype(str)
        counts["type"] = label
        return counts[["items", "n", "type"]]

    @staticmethod
    def _count_other_items(
        data: pd.DataFrame,
        column: str,
        target: Sequence[str],
        item: str,
        label: str,
    ) -> pd.DataFrame:
        count = int((~data[column].astype(str).isin(target)).sum())
        return pd.DataFrame([{"items": item, "n": count, "type": label}])

    def _validate_column(self, column: str) -> None:
        if column not in self._data1.columns:
            msg = f"column {column!r} is missing from data1."
            raise ValueError(msg)
        if column not in self._data2.columns:
            msg = f"column {column!r} is missing from data2."
            raise ValueError(msg)

    @staticmethod
    def _normalize_n(n: int | float) -> int | float:
        if n == float("inf"):
            return n
        limit = int(n)
        if limit < 1:
            msg = "n must be at least 1 or infinity."
            raise ValueError(msg)
        return limit

    @staticmethod
    def _sort_by_abs_contrib(data: pd.DataFrame) -> pd.DataFrame:
        return (
            data.assign(_abs_contrib=data["contrib"].abs())
            .sort_values("_abs_contrib", ascending=False, kind="stable")
            .drop(columns="_abs_contrib")
            .reset_index(drop=True)
        )

    @staticmethod
    def _safe_rate(x: float, n: int) -> float:
        return float(x) / n if n else float("nan")

    def _data_for_column(
        self,
        column: str,
        continuous: ContinuousConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if is_numeric_dtype(self._data1[column]):
            return self._to_factor(column, continuous)
        return self._data1, self._data2

    @staticmethod
    def _summarize_by_column(data: pd.DataFrame, column: str) -> pd.DataFrame:
        grouped = (
            data.groupby(column, observed=True, sort=False)[_OUTCOME_COLUMN]
            .agg(y="sum", n="size")
            .reset_index()
            .rename(columns={column: "items"})
        )
        grouped["rate"] = grouped["y"] / grouped["n"]
        return grouped

    @staticmethod
    def _summarize_info(
        data: pd.DataFrame,
        column: str,
        suffix: str,
    ) -> pd.DataFrame:
        grouped = (
            data.groupby(column, observed=True, sort=False)[_OUTCOME_COLUMN]
            .agg(**{f"x{suffix}": "sum", f"n{suffix}": "size"})
            .reset_index()
            .rename(columns={column: "items"})
        )
        grouped[f"rate{suffix}"] = grouped[f"x{suffix}"] / grouped[f"n{suffix}"]
        return grouped

    @staticmethod
    def _replace_or_append_group(
        base: pd.DataFrame,
        replacement: pd.DataFrame,
        item: Any,
    ) -> pd.DataFrame:
        result = base.copy()
        replacement_row = replacement[replacement["items"] == item]
        mask = result["items"] == item
        if mask.any():
            result.loc[mask, ["y", "n", "rate"]] = replacement_row[
                ["y", "n", "rate"]
            ].to_numpy()
            return result
        return pd.concat([result, replacement_row], ignore_index=True)

    @staticmethod
    def _score_from_summary(data: pd.DataFrame) -> float:
        return float(data["y"].sum() / data["n"].sum())

    @staticmethod
    def _combined_categorical_dtype(
        values1: pd.Series,
        values2: pd.Series,
    ) -> pd.CategoricalDtype | None:
        dtype1 = values1.dtype
        dtype2 = values2.dtype
        if not isinstance(dtype1, pd.CategoricalDtype):
            return None

        categories = list(dtype1.categories)
        if isinstance(dtype2, pd.CategoricalDtype):
            categories.extend(
                category
                for category in dtype2.categories
                if category not in categories
            )
        else:
            categories.extend(value for value in values2 if value not in categories)
        return pd.CategoricalDtype(categories=categories, ordered=dtype1.ordered)

    def _continuous_breaks(
        self,
        column: str,
        continuous: ContinuousConfig,
    ) -> NDArray[np.float64]:
        if continuous.breaks is not None:
            return self._validate_breaks(
                np.asarray(continuous.breaks, dtype=float),
            )

        values = pd.concat(
            [self._data1[column], self._data2[column]],
            ignore_index=True,
        ).astype(float)
        non_missing = values.dropna()
        if non_missing.empty:
            msg = f"column {column!r} must contain at least one non-missing value."
            raise ValueError(msg)

        break_num = continuous.n

        if continuous.split == "width":
            if values.isna().any():
                break_num -= 1
            self._validate_break_num(break_num, continuous)
            breaks = np.linspace(
                non_missing.min(),
                non_missing.max(),
                break_num + 1,
            )
        elif continuous.split == "count":
            self._validate_break_num(
                break_num - int(values.isna().any()),
                continuous,
            )
            breaks = self._compute_breaks(values, break_num)
        else:
            self._validate_break_num(
                break_num - int(values.isna().any()),
                continuous,
            )
            breaks = np.unique(self._compute_breaks(values, break_num * 20))
            data1 = self._data1[[column, _OUTCOME_COLUMN]].dropna(subset=[column])
            data2 = self._data2[[column, _OUTCOME_COLUMN]].dropna(subset=[column])
            while len(breaks) > break_num + 1:
                diff = self._adjacent_rate_diff(data1, data2, column, breaks)
                remove_at = int(np.nanargmin(diff)) + 1
                breaks = np.delete(breaks, remove_at)

        if continuous.pretty:
            pretty = self._pretty_breaks(breaks)
            if len(np.unique(pretty)) < len(pretty):
                warnings.warn(
                    "Prettying breaks reduced the number of breaks. "
                    "Try pretty=False.",
                    stacklevel=2,
                )
                pretty = np.unique(pretty)
            breaks = pretty
        return self._validate_breaks(np.asarray(breaks, dtype=float))

    @staticmethod
    def _validate_break_num(
        break_num: int,
        continuous: ContinuousConfig,
    ) -> None:
        if break_num < 1:
            msg = (
                "continuous.n must leave at least one numeric bin. "
                "Increase n or remove missing values."
            )
            raise ValueError(msg)
        if continuous.split == "rate" and break_num < 2:
            msg = "split='rate' requires at least two numeric bins."
            raise ValueError(msg)

    @staticmethod
    def _validate_breaks(breaks: NDArray[np.float64]) -> NDArray[np.float64]:
        if len(breaks) < _MIN_BREAK_COUNT:
            msg = "continuous breaks must contain at least two values."
            raise ValueError(msg)
        if np.isnan(breaks).any():
            msg = "continuous breaks must not contain NaN."
            raise ValueError(msg)
        if not np.all(np.diff(breaks) > 0):
            msg = "continuous breaks must be strictly increasing."
            raise ValueError(msg)
        return breaks

    @staticmethod
    def _compute_breaks(
        values: pd.Series,
        break_num: int,
    ) -> NDArray[np.float64]:
        if values.isna().any():
            break_num -= 1
        probs = np.linspace(0, 1, break_num + 1)
        return np.asarray(values.quantile(probs).to_numpy(), dtype=float)

    @staticmethod
    def _pretty_breaks(
        breaks: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        result = []
        for value in breaks:
            digits = 0 if value == 0 else np.floor(np.log10(abs(value))) + 1
            base = 10 ** (digits - 2)
            rounded = np.floor(value / base) if value < 0 else np.ceil(value / base)
            result.append(rounded * base)
        return np.asarray(result, dtype=float)

    @staticmethod
    def _adjacent_rate_diff(
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        column: str,
        breaks: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        def summarize(data: pd.DataFrame, name: str) -> pd.Series:
            bins = pd.cut(data[column], bins=breaks, include_lowest=True)
            return (
                data.groupby(bins, observed=True, sort=False)[_OUTCOME_COLUMN]
                .mean()
                .diff(-1)
                .abs()
                .rename(name)
            )

        diff1 = summarize(data1, "diff1")
        diff2 = summarize(data2, "diff2")
        merged = pd.concat([diff1, diff2], axis=1)
        diff = np.sqrt(merged["diff1"] ** 2 + merged["diff2"] ** 2)
        return cast(NDArray[np.float64], np.asarray(diff, dtype=np.float64))

    @staticmethod
    def _cut_to_categorical(
        series: pd.Series,
        breaks: NDArray[np.float64],
    ) -> pd.Series:
        categories = ShipOfTheseus._cut_labels(breaks)
        values = pd.cut(
            series,
            bins=breaks,
            include_lowest=True,
            labels=categories,
        )
        if _MISSING_LABEL not in values.cat.categories:
            values = values.cat.add_categories([_MISSING_LABEL])
        return values.fillna(_MISSING_LABEL)

    @staticmethod
    def _cut_labels(breaks: NDArray[np.float64]) -> list[str]:
        labels = []
        pairs = zip(breaks[:-1], breaks[1:], strict=True)
        for index, (left, right) in enumerate(pairs):
            left_bracket = "[" if index == 0 else "("
            labels.append(
                f"{left_bracket}{ShipOfTheseus._format_break(left)},"
                f"{ShipOfTheseus._format_break(right)}]",
            )
        return labels

    @staticmethod
    def _format_break(value: float) -> str:
        if np.isclose(value, 0.0):
            return "0"
        if np.isclose(value, round(value)):
            return str(int(round(value)))
        return f"{value:.15g}"

    @staticmethod
    def _raise_not_implemented() -> None:
        msg = (
            "TheseusPlot calculation and plotting logic has not been "
            "implemented yet."
        )
        raise NotImplementedError(msg)


def create_ship(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    y: str = "y",
    labels: Sequence[str] = ("Baseline", "Comparison"),
    y_label: str | None = None,
    digits: int = 1,
    text_size: float = 1.0,
    x_label: str | None = None,
) -> ShipOfTheseus:
    """Create a ShipOfTheseus object."""

    return ShipOfTheseus(
        data1=data1,
        data2=data2,
        outcome=y,
        labels=labels,
        y_label=y_label,
        digits=digits,
        text_size=text_size,
        x_label=x_label,
    )
