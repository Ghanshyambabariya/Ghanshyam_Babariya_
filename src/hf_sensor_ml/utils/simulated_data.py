from __future__ import annotations

import numpy as np
import pandas as pd


def generate_grinding_sample(
    *,
    sample_rate_hz: int = 20_000,
    duration_seconds: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    sensor_columns = ["Fx", "Fy", "Fz", "Mz"]
    stage_names = ["stable", "chatter", "wear"]

    samples = sample_rate_hz * duration_seconds
    time_s = np.arange(samples, dtype=np.float32) / sample_rate_hz
    second_bins = np.floor(time_s).astype(int)
    stages = np.array(stage_names)[second_bins % len(stage_names)]

    rng = np.random.default_rng(seed)
    signals = {column: np.zeros(samples, dtype=np.float32) for column in sensor_columns}

    stable_mask = stages == "stable"
    chatter_mask = stages == "chatter"
    wear_mask = stages == "wear"

    signals["Fx"][stable_mask] = 110 + 10 * np.sin(2 * np.pi * 55 * time_s[stable_mask]) + 2.2 * rng.normal(size=stable_mask.sum())
    signals["Fy"][stable_mask] = 78 + 8 * np.sin(2 * np.pi * 35 * time_s[stable_mask] + 0.6) + 1.8 * rng.normal(size=stable_mask.sum())
    signals["Fz"][stable_mask] = 62 + 7 * np.sin(2 * np.pi * 75 * time_s[stable_mask] + 0.2) + 1.5 * rng.normal(size=stable_mask.sum())
    signals["Mz"][stable_mask] = 28 + 3 * np.sin(2 * np.pi * 28 * time_s[stable_mask]) + 0.9 * rng.normal(size=stable_mask.sum())

    signals["Fx"][chatter_mask] = 135 + 14 * np.sin(2 * np.pi * 65 * time_s[chatter_mask]) + 10 * np.sin(2 * np.pi * 420 * time_s[chatter_mask]) + 3.6 * rng.normal(size=chatter_mask.sum())
    signals["Fy"][chatter_mask] = 96 + 12 * np.sin(2 * np.pi * 50 * time_s[chatter_mask]) + 8 * np.sin(2 * np.pi * 380 * time_s[chatter_mask]) + 3.2 * rng.normal(size=chatter_mask.sum())
    signals["Fz"][chatter_mask] = 82 + 11 * np.sin(2 * np.pi * 85 * time_s[chatter_mask]) + 7 * np.sin(2 * np.pi * 460 * time_s[chatter_mask]) + 2.8 * rng.normal(size=chatter_mask.sum())
    signals["Mz"][chatter_mask] = 41 + 4 * np.sin(2 * np.pi * 32 * time_s[chatter_mask]) + 4.5 * np.sin(2 * np.pi * 280 * time_s[chatter_mask]) + 1.4 * rng.normal(size=chatter_mask.sum())

    wear_trend = np.linspace(0, 12, wear_mask.sum(), dtype=np.float32)
    signals["Fx"][wear_mask] = 150 + wear_trend + 9 * np.sin(2 * np.pi * 45 * time_s[wear_mask]) + 2.6 * rng.normal(size=wear_mask.sum())
    signals["Fy"][wear_mask] = 109 + 0.7 * wear_trend + 7 * np.sin(2 * np.pi * 30 * time_s[wear_mask]) + 2.2 * rng.normal(size=wear_mask.sum())
    signals["Fz"][wear_mask] = 90 + 0.5 * wear_trend + 6 * np.sin(2 * np.pi * 60 * time_s[wear_mask]) + 2.0 * rng.normal(size=wear_mask.sum())
    signals["Mz"][wear_mask] = 48 + 0.35 * wear_trend + 3 * np.sin(2 * np.pi * 24 * time_s[wear_mask]) + 1.1 * rng.normal(size=wear_mask.sum())

    return pd.DataFrame(
        {
            "time_s": time_s,
            "Fx": signals["Fx"],
            "Fy": signals["Fy"],
            "Fz": signals["Fz"],
            "Mz": signals["Mz"],
            "stage": stages,
        }
    )
