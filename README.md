# Ghanshyam_Babariya_

Machine learning project for high-frequency grinding sensor data.

## What Happens Now

- If `data/raw/grinding_20khz.csv` does not exist, the pipeline generates a realistic simulated 20 kHz grinding file for `Fx`, `Fy`, `Fz`, and `Mz` over 20 seconds.
- If you replace that CSV with your own file, the pipeline uses your file instead.
- The outputs are rewritten each time the pipeline runs, so results stay current with your latest data and code.

## Expected Input Format

The CSV should contain:

- `time_s`
- `Fx`
- `Fy`
- `Fz`
- `Mz`
- `stage` for supervised learning

## Run Locally

```powershell
C:\Users\ghans\AppData\Local\Programs\Python\Python312\python.exe scripts\run_real_pipeline.py
```

## Generated Outputs

- `data/raw/grinding_20khz.csv`
- `data/processed/grinding_window_features.csv`
- `reports/figures/raw_signal_preview.png`
- `reports/figures/top_features.png`
- `reports/real_run_results.json`

## GitHub Updates

A GitHub Actions workflow reruns the pipeline on pushes that change the data, config, scripts, or source code and commits refreshed output files back to the repository.
