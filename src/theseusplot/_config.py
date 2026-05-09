"""Configuration objects for TheseusPlot.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SplitMethod = Literal["count", "width", "rate"]


@dataclass(frozen=True)
class ContinuousConfig:
    """Configuration for discretizing continuous variables."""

    n: int = 10
    pretty: bool = True
    split: SplitMethod = "count"
    breaks: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.n <= 0:
            msg = "n must be a positive integer."
            raise ValueError(msg)
        if self.split not in {"count", "width", "rate"}:
            msg = "split must be one of 'count', 'width', or 'rate'."
            raise ValueError(msg)


def continuous_config(
    n: int = 10,
    pretty: bool = True,
    split: SplitMethod = "count",
    breaks: list[float] | tuple[float, ...] | None = None,
) -> ContinuousConfig:
    """Create a continuous variable configuration."""

    normalized_breaks = tuple(breaks) if breaks is not None else None
    return ContinuousConfig(
        n=n,
        pretty=pretty,
        split=split,
        breaks=normalized_breaks,
    )
