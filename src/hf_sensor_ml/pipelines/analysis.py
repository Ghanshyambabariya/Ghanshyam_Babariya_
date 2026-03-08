"""Exploratory analysis helpers for high-frequency sensor data."""


def analysis_checklist() -> list[str]:
    return [
        "Validate timestamp monotonicity and missing intervals.",
        "Measure per-sensor drift, spikes, and noise floors.",
        "Profile class balance or target coverage over time.",
        "Inspect correlations across channels and sensors.",
    ]
