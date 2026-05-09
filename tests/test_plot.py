import pandas as pd
import pytest

from theseusplot import create_ship


def _require_pyplot():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    return pytest.importorskip("matplotlib.pyplot")


def test_plot_returns_matplotlib_figure_and_axes() -> None:
    plt = _require_pyplot()

    data1 = pd.DataFrame({"group": ["A", "A", "B", "B"], "y": [1, 1, 1, 1]})
    data2 = pd.DataFrame({"group": ["A", "A", "B", "B"], "y": [0, 0, 1, 1]})
    ship = create_ship(data1, data2, labels=("Before", "After"), y_label="Rate")

    fig, ax = ship.plot("group")

    try:
        assert ax.get_title() == "group"
        assert ax.get_ylabel() == "Rate"
        assert [tick.get_text() for tick in ax.get_xticklabels()] == [
            "Before",
            "A",
            "B",
            "After",
        ]
        assert fig is ax.figure
    finally:
        plt.close(fig)


def test_plot_respects_explicit_levels() -> None:
    plt = _require_pyplot()

    data1 = pd.DataFrame({"group": ["A", "B"], "y": [1, 1]})
    data2 = pd.DataFrame({"group": ["A", "B"], "y": [0, 1]})
    ship = create_ship(data1, data2)

    fig, ax = ship.plot("group", levels=["B", "A"])

    try:
        assert [tick.get_text() for tick in ax.get_xticklabels()] == [
            "Original",
            "B",
            "A",
            "Refitted",
        ]
    finally:
        plt.close(fig)


def test_plot_validates_main_item() -> None:
    _require_pyplot()

    data1 = pd.DataFrame({"group": ["A", "B"], "y": [1, 1]})
    data2 = pd.DataFrame({"group": ["A", "B"], "y": [0, 1]})
    ship = create_ship(data1, data2)

    with pytest.raises(ValueError, match="main_item"):
        ship.plot("group", main_item="C")


def test_plot_flip_returns_horizontal_matplotlib_plot() -> None:
    plt = _require_pyplot()

    data1 = pd.DataFrame({"group": ["A", "A", "B", "B"], "y": [1, 1, 1, 1]})
    data2 = pd.DataFrame({"group": ["A", "A", "B", "B"], "y": [0, 0, 1, 1]})
    ship = create_ship(data1, data2, labels=("Before", "After"), y_label="Rate")

    fig, ax = ship.plot_flip("group")

    try:
        assert ax.get_title() == "group"
        assert ax.get_xlabel() == "Rate"
        assert [tick.get_text() for tick in ax.get_yticklabels()] == [
            "After",
            "B",
            "A",
            "Before",
        ]
        assert fig is ax.figure
    finally:
        plt.close(fig)


def test_plot_flip_respects_reversed_explicit_levels() -> None:
    plt = _require_pyplot()

    data1 = pd.DataFrame({"group": ["A", "B"], "y": [1, 1]})
    data2 = pd.DataFrame({"group": ["A", "B"], "y": [0, 1]})
    ship = create_ship(data1, data2)

    fig, ax = ship.plot_flip("group", levels=["B", "A"])

    try:
        assert [tick.get_text() for tick in ax.get_yticklabels()] == [
            "Refitted",
            "A",
            "B",
            "Original",
        ]
    finally:
        plt.close(fig)


def test_plot_flip_validates_main_item() -> None:
    _require_pyplot()

    data1 = pd.DataFrame({"group": ["A", "B"], "y": [1, 1]})
    data2 = pd.DataFrame({"group": ["A", "B"], "y": [0, 1]})
    ship = create_ship(data1, data2)

    with pytest.raises(ValueError, match="main_item"):
        ship.plot_flip("group", main_item="C")
