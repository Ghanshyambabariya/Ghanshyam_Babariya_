"""Exploratory analysis helpers for high-frequency sensor data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def validate_input_frame(df: pd.DataFrame, sensor_columns: list[str], timestamp_column: str) -> None:
    required = [timestamp_column, *sensor_columns]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[timestamp_column].isna().any():
        raise ValueError(f"Timestamp column '{timestamp_column}' contains missing values")


def summarize_raw_data(
    df: pd.DataFrame,
    *,
    sample_rate_hz: int,
    sensor_columns: list[str],
    timestamp_column: str,
    label_column: str | None,
) -> dict:
    summary: dict[str, object] = {
        "sample_rate_hz": sample_rate_hz,
        "duration_seconds": round(len(df) / sample_rate_hz, 3),
        "num_samples": int(len(df)),
        "timestamp_start": float(df[timestamp_column].iloc[0]),
        "timestamp_end": float(df[timestamp_column].iloc[-1]),
        "channel_summary": {},
    }
    if label_column and label_column in df.columns:
        summary["label_distribution"] = df[label_column].value_counts().sort_index().to_dict()
    for column in sensor_columns:
        summary["channel_summary"][column] = {
            "mean": round(float(df[column].mean()), 4),
            "std": round(float(df[column].std()), 4),
            "min": round(float(df[column].min()), 4),
            "max": round(float(df[column].max()), 4),
        }
    return summary


def save_signal_preview(
    df: pd.DataFrame,
    output_path: Path,
    *,
    sample_rate_hz: int,
    sensor_columns: list[str],
    timestamp_column: str,
    seconds: float = 0.2,
) -> None:
    sample_count = int(sample_rate_hz * seconds)
    preview = df.iloc[:sample_count]
    fig, axes = plt.subplots(len(sensor_columns), 1, figsize=(12, 2.2 * len(sensor_columns)), sharex=True)
    if len(sensor_columns) == 1:
        axes = [axes]
    for axis, column in zip(axes, sensor_columns, strict=True):
        axis.plot(preview[timestamp_column], preview[column], linewidth=0.8)
        axis.set_ylabel(column)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Signal preview")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
