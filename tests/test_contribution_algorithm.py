import math

import pandas as pd

from theseusplot import continuous_config, create_ship


def test_replacement_amounts_follow_bidirectional_theseus_algorithm() -> None:
    data1 = pd.DataFrame({"group": ["A", "B"], "y": [1, 1]})
    data2 = pd.DataFrame({"group": ["A", "B"], "y": [0, 1]})
    ship = create_ship(data1, data2)

    grouped1 = ship._summarize_by_column(ship._data1, "group")
    grouped2 = ship._summarize_by_column(ship._data2, "group")
    score1, score2 = ship._compute_scores("group")

    amounts = ship._compute_replacement_amounts(
        original=grouped1,
        refitted=grouped2,
        original_score=score1,
        refitted_score=score2,
    )

    assert amounts["items"].tolist() == ["A", "B", "A", "B"]
    assert amounts["amount"].tolist() == [-0.5, 0.0, -0.5, 0.0]


def test_contributions_are_scaled_to_overall_difference() -> None:
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

    contribution = ship._compute_contribution("group", continuous_config())

    assert set(contribution["items"]) == {"A", "B", "C", "(Missing)"}
    assert math.isclose(
        float(contribution["contrib"].sum()),
        data2["y"].mean() - data1["y"].mean(),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_average_replacement_amounts_preserves_categorical_order() -> None:
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

    contribution = ship._compute_contribution("segment", continuous_config())

    assert contribution["items"].tolist() == levels
