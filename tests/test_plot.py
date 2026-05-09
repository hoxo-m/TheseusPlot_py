import pandas as pd
import pytest

from theseusplot import create_ship


def _require_pyplot():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    return pytest.importorskip("matplotlib.pyplot")


def _waterfall_colors(ax) -> list[str]:
    colors = pytest.importorskip("matplotlib.colors")
    return [
        colors.to_hex(patch.get_facecolor())
        for patch in ax.patches
        if patch.get_zorder() == 3
    ]


def _size_bar_styles(ax) -> list[tuple[str, float]]:
    colors = pytest.importorskip("matplotlib.colors")
    return [
        (colors.to_hex(patch.get_facecolor()), patch.get_facecolor()[3])
        for patch in ax.patches
        if patch.get_zorder() == 1
    ]


def _visible_yticklabels(ax) -> list[str]:
    labels = []
    for tick in ax.get_yticklabels():
        _, display_y = ax.transData.transform((0, tick.get_position()[1]))
        labels.append((display_y, tick.get_text()))
    return [label for _, label in sorted(labels, reverse=True)]


def _plot_texts(ax) -> list[str]:
    return [text.get_text() for text in ax.texts]


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


def test_plot_colors_negative_contributions_red_and_positive_blue() -> None:
    plt = _require_pyplot()

    data1 = pd.DataFrame(
        {"group": ["A"] * 10 + ["B"] * 10, "y": [0] * 10 + [1] * 10},
    )
    data2 = pd.DataFrame(
        {"group": ["A"] * 10 + ["B"] * 30, "y": [1] * 10 + [0] * 30},
    )
    ship = create_ship(data1, data2)

    fig, ax = ship.plot("group")

    try:
        assert _waterfall_colors(ax) == [
            "#00bfc4",
            "#f8766d",
            "#00bfc4",
            "#00bfc4",
        ]
    finally:
        plt.close(fig)


def test_plot_size_bars_use_r_colors_without_alpha() -> None:
    plt = _require_pyplot()

    data1 = pd.DataFrame({"group": ["A", "B"], "y": [1, 1]})
    data2 = pd.DataFrame({"group": ["A", "B"], "y": [0, 1]})
    ship = create_ship(data1, data2)

    fig, ax = ship.plot("group")

    try:
        assert _size_bar_styles(ax) == [
            ("#7cae00", 1.0),
            ("#7cae00", 1.0),
            ("#c77cff", 1.0),
            ("#c77cff", 1.0),
        ]
    finally:
        plt.close(fig)


def test_plot_flip_returns_horizontal_matplotlib_plot() -> None:
    plt = _require_pyplot()

    data1 = pd.DataFrame({"group": ["A", "A", "B", "B"], "y": [1, 1, 1, 1]})
    data2 = pd.DataFrame({"group": ["A", "A", "B", "B"], "y": [0, 0, 1, 1]})
    ship = create_ship(data1, data2, labels=("Before", "After"), y_label="Rate")

    fig, ax = ship.plot_flip("group")

    try:
        assert ax.get_title() == "group"
        assert ax.get_xlabel() == "Rate"
        assert _visible_yticklabels(ax) == [
            "Before",
            "A",
            "B",
            "After",
        ]
        assert fig is ax.figure
    finally:
        plt.close(fig)


def test_plot_flip_colors_original_negative_contributions_red() -> None:
    plt = _require_pyplot()

    data1 = pd.DataFrame(
        {"group": ["A"] * 10 + ["B"] * 10, "y": [0] * 10 + [1] * 10},
    )
    data2 = pd.DataFrame(
        {"group": ["A"] * 10 + ["B"] * 30, "y": [1] * 10 + [0] * 30},
    )
    ship = create_ship(data1, data2)

    fig, ax = ship.plot_flip("group")

    try:
        assert _waterfall_colors(ax) == [
            "#00bfc4",
            "#00bfc4",
            "#f8766d",
            "#00bfc4",
        ]
    finally:
        plt.close(fig)


def test_plot_flip_labels_use_original_contribution_signs() -> None:
    plt = _require_pyplot()

    data1 = pd.DataFrame(
        {"group": ["A"] * 10 + ["B"] * 10, "y": [0] * 10 + [1] * 10},
    )
    data2 = pd.DataFrame(
        {"group": ["A"] * 10 + ["B"] * 30, "y": [1] * 10 + [0] * 30},
    )
    ship = create_ship(data1, data2)

    fig, ax = ship.plot_flip("group")

    try:
        assert _plot_texts(ax) == ["25", "37.5", "-62.5", "50"]
    finally:
        plt.close(fig)


def test_plot_flip_size_bars_use_r_colors_without_alpha() -> None:
    plt = _require_pyplot()

    data1 = pd.DataFrame({"group": ["A", "B"], "y": [1, 1]})
    data2 = pd.DataFrame({"group": ["A", "B"], "y": [0, 1]})
    ship = create_ship(data1, data2)

    fig, ax = ship.plot_flip("group")

    try:
        assert _size_bar_styles(ax) == [
            ("#7cae00", 1.0),
            ("#7cae00", 1.0),
            ("#c77cff", 1.0),
            ("#c77cff", 1.0),
        ]
    finally:
        plt.close(fig)


def test_plot_flip_respects_reversed_explicit_levels() -> None:
    plt = _require_pyplot()

    data1 = pd.DataFrame({"group": ["A", "B"], "y": [1, 1]})
    data2 = pd.DataFrame({"group": ["A", "B"], "y": [0, 1]})
    ship = create_ship(data1, data2)

    fig, ax = ship.plot_flip("group", levels=["B", "A"])

    try:
        assert _visible_yticklabels(ax) == [
            "Original",
            "B",
            "A",
            "Refitted",
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
