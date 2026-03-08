from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hf_sensor_ml.pipelines.analysis import save_signal_preview, summarize_raw_data, validate_input_frame
from hf_sensor_ml.pipelines.classical_ml import train_baseline_models
from hf_sensor_ml.pipelines.feature_engineering import build_feature_dataset
from hf_sensor_ml.pipelines.sequence_models import train_sequence_models
from hf_sensor_ml.utils.simulated_data import generate_grinding_sample


def save_top_feature_plot(results: dict, output_path: Path) -> None:
    top_features = results["models"]["random_forest"]["top_features"]
    labels = [item["feature"] for item in top_features][::-1]
    values = [item["importance"] for item in top_features][::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, values)
    ax.set_title("Top random-forest features")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_or_create_input(repo_root: Path, config: dict) -> tuple[pd.DataFrame, Path, str]:
    raw_data_path = repo_root / config["raw_data_path"]
    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_data_path.exists():
        return pd.read_csv(raw_data_path), raw_data_path, "existing"

    if not config.get("generate_sample_if_missing", False):
        raise FileNotFoundError(
            f"Input file not found: {raw_data_path}. Put your grinding CSV there first."
        )

    df = generate_grinding_sample(
        sample_rate_hz=config["sample_rate_hz"],
        duration_seconds=config.get("duration_seconds", 20),
        seed=config.get("sample_seed", 42),
    )
    df.to_csv(raw_data_path, index=False)
    return df, raw_data_path, "generated"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads((repo_root / "configs" / "grinding_20khz.json").read_text(encoding="utf-8"))

    df, raw_data_path, input_mode = load_or_create_input(repo_root, config)
    validate_input_frame(df, config["sensor_columns"], config["timestamp_column"])

    analysis_summary = summarize_raw_data(
        df,
        sample_rate_hz=config["sample_rate_hz"],
        sensor_columns=config["sensor_columns"],
        timestamp_column=config["timestamp_column"],
        label_column=config.get("label_column"),
    )

    preview_plot_path = repo_root / "reports" / "figures" / "raw_signal_preview.png"
    save_signal_preview(
        df,
        preview_plot_path,
        sample_rate_hz=config["sample_rate_hz"],
        sensor_columns=config["sensor_columns"],
        timestamp_column=config["timestamp_column"],
    )

    feature_df = build_feature_dataset(
        df,
        sample_rate_hz=config["sample_rate_hz"],
        sensor_columns=config["sensor_columns"],
        timestamp_column=config["timestamp_column"],
        label_column=config.get("label_column"),
        window_seconds=config["feature_window_seconds"],
        stride_seconds=config["feature_stride_seconds"],
    )
    feature_data_path = repo_root / "data" / "processed" / "grinding_window_features.csv"
    feature_data_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(feature_data_path, index=False)

    label_column = config.get("label_column")
    if not label_column or label_column not in df.columns:
        raise ValueError(
            "A label column is required for baseline ML and sequence models. "
            "Add one such as 'stage' or update configs/grinding_20khz.json."
        )

    baseline_results = train_baseline_models(feature_df, label_column=label_column)
    feature_plot_path = repo_root / "reports" / "figures" / "top_features.png"
    save_top_feature_plot(baseline_results, feature_plot_path)

    sequence_results = train_sequence_models(
        df,
        sample_rate_hz=config["sample_rate_hz"],
        sensor_columns=config["sensor_columns"],
        label_column=label_column,
        window_seconds=config["sequence_window_seconds"],
        stride_seconds=config["sequence_stride_seconds"],
        downsample_factor=config["sequence_downsample_factor"],
    )

    results = {
        "input_mode": input_mode,
        "analysis": analysis_summary,
        "feature_engineering": {
            "num_feature_windows": int(len(feature_df)),
            "num_features": int(feature_df.drop(columns=[label_column]).shape[1]),
        },
        "baseline_models": baseline_results,
        "sequence_models": sequence_results,
        "artifacts": {
            "raw_data_path": str(raw_data_path),
            "feature_data_path": str(feature_data_path),
            "preview_plot": str(preview_plot_path),
            "feature_plot": str(feature_plot_path),
        },
    }

    results_path = repo_root / "reports" / "real_run_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
