import pandas as pd
import pytest

from theseusplot import (
    ContinuousConfig,
    ShipOfTheseus,
    continuous_config,
    create_ship,
)


def test_continuous_config_defaults() -> None:
    config = continuous_config()

    assert config == ContinuousConfig(
        n=10,
        pretty=True,
        split="count",
        breaks=None,
    )


def test_continuous_config_normalizes_breaks() -> None:
    config = continuous_config(n=3, pretty=False, split="width", breaks=[0, 1, 2])

    assert config.breaks == (0, 1, 2)


def test_continuous_config_validates_n() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        continuous_config(n=0)


def test_create_ship_returns_ship_object() -> None:
    data1 = pd.DataFrame({"group": ["A", "B"], "y": [1, 0]})
    data2 = pd.DataFrame({"group": ["A", "B"], "y": [0, 1]})

    ship = create_ship(
        data1,
        data2,
        y="y",
        labels=("Before", "After"),
        x_label="Segment",
    )

    assert isinstance(ship, ShipOfTheseus)
    assert ship.outcome == "y"
    assert ship.labels == ("Before", "After")
    assert ship.x_label == "Segment"


def test_create_ship_uses_r_0_3_0_defaults() -> None:
    data1 = pd.DataFrame({"group": ["A"], "y": [1]})
    data2 = pd.DataFrame({"group": ["A"], "y": [0]})

    ship = create_ship(data1, data2)

    assert ship.labels == ("Baseline", "Comparison")
    assert ship.digits == 1


def test_create_ship_validates_outcome_column() -> None:
    data1 = pd.DataFrame({"group": ["A"], "y": [1]})
    data2 = pd.DataFrame({"group": ["A"], "z": [1]})

    with pytest.raises(ValueError, match="missing from data2"):
        create_ship(data1, data2, y="y")


def test_create_ship_validates_label_count() -> None:
    data1 = pd.DataFrame({"group": ["A"], "y": [1]})
    data2 = pd.DataFrame({"group": ["A"], "y": [0]})

    with pytest.raises(ValueError, match="exactly two"):
        create_ship(data1, data2, labels=("Only one",))


def test_create_ship_validates_missing_outcome_values() -> None:
    data1 = pd.DataFrame({"group": ["A"], "y": [None]})
    data2 = pd.DataFrame({"group": ["A"], "y": [0]})

    with pytest.raises(ValueError, match="must not contain missing"):
        create_ship(data1, data2)


def test_table_returns_basic_contribution_table() -> None:
    data1 = pd.DataFrame({"group": ["A", "B"], "y": [1, 1]})
    data2 = pd.DataFrame({"group": ["A", "B"], "y": [0, 1]})
    ship = create_ship(data1, data2)

    table = ship.table("group")

    assert list(table.columns) == [
        "group",
        "contrib",
        "n1",
        "n2",
        "x1",
        "x2",
        "rate1",
        "rate2",
    ]
    assert table["group"].tolist() == ["A", "B"]
    assert table["contrib"].tolist() == [-0.5, 0.0]


def test_table_validates_column_name() -> None:
    data1 = pd.DataFrame({"group": ["A"], "y": [1]})
    data2 = pd.DataFrame({"group": ["A"], "y": [0]})
    ship = create_ship(data1, data2)

    with pytest.raises(ValueError, match="missing from data1"):
        ship.table("missing")


def test_table_validates_n() -> None:
    data1 = pd.DataFrame({"group": ["A"], "y": [1]})
    data2 = pd.DataFrame({"group": ["A"], "y": [0]})
    ship = create_ship(data1, data2)

    with pytest.raises(ValueError, match="at least 1"):
        ship.table("group", n=0)
