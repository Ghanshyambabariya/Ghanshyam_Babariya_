"""Feature engineering for windowed sensor data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _dominant_frequency(values: np.ndarray, sample_rate_hz: int) -> float:
    centered = values - values.mean()
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(len(centered), d=1 / sample_rate_hz)
    if len(freqs) <= 1:
        return 0.0
    idx = int(np.argmax(np.abs(spectrum[1:])) + 1)
    return float(freqs[idx])


def build_feature_dataset(
    df: pd.DataFrame,
    *,
    sample_rate_hz: int,
    sensor_columns: list[str],
    timestamp_column: str,
    label_column: str | None,
    window_seconds: float,
    stride_seconds: float,
) -> pd.DataFrame:
    window_size = int(window_seconds * sample_rate_hz)
    stride = int(stride_seconds * sample_rate_hz)
    records: list[dict[str, float | str]] = []

    for start in range(0, len(df) - window_size + 1, stride):
        window = df.iloc[start : start + window_size]
        record: dict[str, float | str] = {
            "window_start_s": float(window[timestamp_column].iloc[0]),
            "window_end_s": float(window[timestamp_column].iloc[-1]),
        }
        if label_column and label_column in window.columns:
            record[label_column] = str(window[label_column].mode().iloc[0])
        for column in sensor_columns:
            values = window[column].to_numpy(dtype=float)
            rms = float(np.sqrt(np.mean(np.square(values))))
            record[f"{column}_mean"] = float(np.mean(values))
            record[f"{column}_std"] = float(np.std(values))
            record[f"{column}_rms"] = rms
            record[f"{column}_ptp"] = float(np.ptp(values))
            record[f"{column}_dominant_freq_hz"] = _dominant_frequency(values, sample_rate_hz)
        if len(sensor_columns) >= 3:
            cols = sensor_columns[:3]
            record["resultant_mean"] = float(
                np.mean(np.sqrt(sum(np.square(window[col]) for col in cols)))
            )
        if len(sensor_columns) >= 2:
            record[f"{sensor_columns[0]}_{sensor_columns[1]}_corr"] = float(
                window[sensor_columns[:2]].corr().iloc[0, 1]
            )
        records.append(record)

    return pd.DataFrame.from_records(records)
