import math

import numpy as np
import pandas as pd
import pytest

from theseusplot import continuous_config, create_ship


def test_count_split_uses_quantile_breaks() -> None:
    data1 = pd.DataFrame({"x": [0, 1], "y": [1, 1]})
    data2 = pd.DataFrame({"x": [2, 3], "y": [0, 0]})
    ship = create_ship(data1, data2)

    table = ship.table("x", continuous=continuous_config(n=2, pretty=False))

    assert table["x"].tolist() == ["[0,1.5]", "(1.5,3]"]
    assert table["n1"].tolist() == [2, 0]
    assert table["n2"].tolist() == [0, 2]
    assert math.isclose(float(table["contrib"].sum()), -1.0, abs_tol=1e-12)


def test_width_split_reduces_numeric_bins_when_values_are_missing() -> None:
    data1 = pd.DataFrame({"x": [0.0, 1.0, np.nan, 3.0], "y": [1, 1, 0, 0]})
    data2 = pd.DataFrame({"x": [0.0, 2.0, 3.0, np.nan], "y": [1, 0, 0, 1]})
    ship = create_ship(data1, data2)

    table = ship.table(
        "x",
        continuous=continuous_config(n=3, pretty=False, split="width"),
    )

    assert table["x"].tolist() == ["[0,1.5]", "(1.5,3]", "(Missing)"]
    assert table["n1"].tolist() == [2, 1, 1]
    assert table["n2"].tolist() == [1, 2, 1]


def test_custom_breaks_override_automatic_binning() -> None:
    data1 = pd.DataFrame({"x": [0, 1, 2], "y": [1, 1, 0]})
    data2 = pd.DataFrame({"x": [2, 3, 4], "y": [1, 0, 0]})
    ship = create_ship(data1, data2)

    table = ship.table(
        "x",
        continuous=continuous_config(breaks=[0, 2, 4], pretty=True),
    )

    assert table["x"].tolist() == ["[0,2]", "(2,4]"]


def test_rate_split_returns_requested_number_of_numeric_bins() -> None:
    data1 = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5],
            "y": [1, 1, 1, 0, 0, 0],
        },
    )
    data2 = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5],
            "y": [1, 0, 0, 0, 0, 0],
        },
    )
    ship = create_ship(data1, data2)

    table = ship.table(
        "x",
        continuous=continuous_config(n=3, pretty=False, split="rate"),
    )

    assert len(table) == 3
    assert math.isclose(
        float(table["contrib"].sum()),
        data2["y"].mean() - data1["y"].mean(),
        abs_tol=1e-12,
    )


def test_custom_breaks_must_be_strictly_increasing() -> None:
    data1 = pd.DataFrame({"x": [0, 1], "y": [1, 0]})
    data2 = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    ship = create_ship(data1, data2)

    with pytest.raises(ValueError, match="strictly increasing"):
        ship.table("x", continuous=continuous_config(breaks=[0, 1, 1]))


def test_missing_values_need_at_least_one_numeric_bin() -> None:
    data1 = pd.DataFrame({"x": [0.0, np.nan], "y": [1, 0]})
    data2 = pd.DataFrame({"x": [1.0, np.nan], "y": [0, 1]})
    ship = create_ship(data1, data2)

    with pytest.raises(ValueError, match="at least one numeric bin"):
        ship.table("x", continuous=continuous_config(n=1, pretty=False))
