# Ghanshyam_Babariya_

Machine learning project scaffold for high-frequency sensor data.

## Workflow

1. Analysis: inspect sensor quality, drift, sampling gaps, outliers, and label alignment.
2. Feature engineering: create rolling, spectral, statistical, and window-based features.
3. ML models: train baseline tree-based and linear models on engineered features.
4. Advanced sequence models: train LSTM and time-series architectures on ordered sensor windows.

## Project Structure

```text
.
|-- configs/
|-- data/
|   |-- raw/
|   |-- interim/
|   |-- processed/
|   `-- external/
|-- models/
|-- notebooks/
|   |-- 01_analysis/
|   |-- 02_feature_engineering/
|   |-- 03_ml_models/
|   `-- 04_sequence_models/
|-- reports/
|   `-- figures/
`-- src/
    `-- hf_sensor_ml/
        |-- config.py
        |-- pipelines/
        `-- utils/
```

## Suggested Order

- Put original sensor exports in `data/raw/`.
- Use `notebooks/01_analysis/` for exploratory analysis and quality checks.
- Convert reusable logic into `src/hf_sensor_ml/pipelines/`.
- Save trained artifacts in `models/`.
- Store plots and evaluation outputs in `reports/figures/`.

## Next Steps

- Define the prediction target and sensor sampling rate.
- Add a schema for timestamps, sensor IDs, channels, and labels.
- Start with descriptive analysis before selecting features or model families.
