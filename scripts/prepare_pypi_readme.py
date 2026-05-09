"""Prepare README image URLs for PyPI rendering."""

from __future__ import annotations

from pathlib import Path

RAW_BASE_URL = "https://raw.githubusercontent.com/hoxo-m/TheseusPlot_py/main/"
README_PATH = Path("README.md")


def main() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    readme = readme.replace(
        'src="README-figures/',
        f'src="{RAW_BASE_URL}README-figures/',
    )
    README_PATH.write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
