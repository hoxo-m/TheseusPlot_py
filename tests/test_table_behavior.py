import math

import pandas as pd

from theseusplot import create_ship


def test_table_keeps_missing_and_asymmetric_categories() -> None:
    data1 = pd.DataFrame(
        {
            "group": ["A", "A", "B", None],
            "y": [1, 1, 0, 1],
        },
    )
    data2 = pd.DataFrame(
        {
            "group": ["A", "C", "C", None],
            "y": [0, 1, 0, 0],
        },
    )
    ship = create_ship(data1, data2)

    table = ship.table("group")

    assert set(table["group"]) == {"A", "B", "C", "(Missing)"}
    assert table.loc[table["group"] == "C", "n1"].item() == 0
    assert table.loc[table["group"] == "C", "x1"].item() == 0
    assert table.loc[table["group"] == "B", "n2"].item() == 0
    assert table.loc[table["group"] == "B", "x2"].item() == 0
    assert math.isclose(
        float(table["contrib"].sum()),
        data2["y"].mean() - data1["y"].mean(),
        abs_tol=1e-12,
    )


def test_table_preserves_ordered_categorical_order() -> None:
    levels = ["Low", "Medium", "High"]
    data1 = pd.DataFrame(
        {
            "segment": pd.Categorical(
                ["Low", "Low", "Medium", "Medium", "High", "High"],
                categories=levels,
                ordered=True,
            ),
            "y": [1, 1, 1, 0, 1, 1],
        },
    )
    data2 = pd.DataFrame(
        {
            "segment": pd.Categorical(
                ["Low", "Low", "Medium", "Medium", "High", "High"],
                categories=levels,
                ordered=True,
            ),
            "y": [1, 0, 1, 1, 0, 0],
        },
    )
    ship = create_ship(data1, data2)

    table = ship.table("segment")

    assert table["segment"].tolist() == levels


def test_table_groups_low_contribution_rows_into_other_bucket() -> None:
    data1 = pd.DataFrame(
        {
            "category": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
            "y": [1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
        },
    )
    data2 = pd.DataFrame(
        {
            "category": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
            "y": [0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
        },
    )
    ship = create_ship(data1, data2)

    table = ship.table("category", n=3)

    assert table["category"].tolist() == ["A", "B", "Sum of 3 other attributes"]
    assert math.isclose(float(table["contrib"].sum()), -0.1, abs_tol=1e-12)
    other_row = table[table["category"] == "Sum of 3 other attributes"]
    assert other_row["n1"].item() == 6
    assert other_row["n2"].item() == 6
