# Ghanshyam_Babariya_

Machine learning project for high-frequency grinding sensor data.

## Expected Real Input

Place the real dataset at `data/raw/grinding_20khz.csv` with these columns:

- `time_s`
- `Fx`
- `Fy`
- `Fz`
- `Mz`
- `stage` for supervised learning

`stage` is required if you want baseline ML and LSTM / temporal CNN results.

## Pipeline Order

1. Analysis of the raw 20 kHz signals
2. Feature engineering on fixed windows
3. Baseline ML on engineered features
4. Advanced sequence models on ordered sensor windows

## Run

Use:

```powershell
C:\Users\ghans\AppData\Local\Programs\Python\Python312\python.exe scripts\run_real_pipeline.py
```

## Outputs

- `data/processed/grinding_window_features.csv`
- `reports/figures/raw_signal_preview.png`
- `reports/figures/top_features.png`
- `reports/real_run_results.json`
