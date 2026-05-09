import json
import math
from pathlib import Path

import pandas as pd
import pytest

from theseusplot import create_ship

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _fixture_paths() -> list[Path]:
    return sorted(FIXTURE_DIR.glob("*.json"))


def _data_frame(
    fixture: dict[str, object],
    key: str,
) -> pd.DataFrame:
    spec = fixture[key]
    assert isinstance(spec, dict)
    data = pd.DataFrame(spec["rows"], columns=spec["columns"])

    categorical_levels = fixture.get("categorical_levels", {})
    assert isinstance(categorical_levels, dict)
    for column, levels in categorical_levels.items():
        data[column] = pd.Categorical(data[column], categories=levels, ordered=True)
    return data


def _assert_table_matches_fixture(
    actual: pd.DataFrame,
    expected_table: dict[str, object],
) -> None:
    assert list(actual.columns) == expected_table["columns"]

    expected_rows = expected_table["rows"]
    assert isinstance(expected_rows, list)
    assert len(actual) == len(expected_rows)

    for (_, actual_row), expected_row in zip(
        actual.iterrows(),
        expected_rows,
        strict=True,
    ):
        assert isinstance(expected_row, dict)
        for column, expected_value in expected_row.items():
            actual_value = actual_row[column]
            if expected_value is None:
                assert pd.isna(actual_value)
            elif isinstance(expected_value, float):
                assert math.isclose(
                    float(actual_value),
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            else:
                assert actual_value == expected_value


def test_table_fixtures_exist() -> None:
    assert _fixture_paths()


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=lambda path: path.stem)
def test_table_fixture_schema_and_totals(fixture_path: Path) -> None:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    expected_table = fixture["expected_table"]
    columns = expected_table["columns"]
    rows = expected_table["rows"]

    assert fixture["source"]["package"] == "TheseusPlot"
    assert columns[1:] == ["contrib", "n1", "n2", "x1", "x2", "rate1", "rate2"]
    assert rows

    row_keys = set(columns)
    for row in rows:
        assert set(row) == row_keys

    total_contrib = sum(row["contrib"] for row in rows)
    assert math.isclose(
        total_contrib,
        expected_table["overall_diff"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=lambda path: path.stem)
def test_table_matches_r_fixture(fixture_path: Path) -> None:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    call = fixture["call"]
    assert isinstance(call, dict)
    expected_table = fixture["expected_table"]
    assert isinstance(expected_table, dict)

    ship = create_ship(
        _data_frame(fixture, "data1"),
        _data_frame(fixture, "data2"),
        y="y",
    )
    n = float("inf") if call["n"] is None else int(call["n"])

    actual = ship.table(str(call["column"]), n=n)

    _assert_table_matches_fixture(actual, expected_table)
