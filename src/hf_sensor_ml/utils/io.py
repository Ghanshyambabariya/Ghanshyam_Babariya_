"""Helpers for loading and validating sensor datasets."""

from pathlib import Path
import pandas as pd


def load_csv(path: str | Path, *, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path, parse_dates=parse_dates)
