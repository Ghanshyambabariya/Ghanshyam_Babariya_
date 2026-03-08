"""Feature engineering ideas for windowed sensor data."""


def feature_families() -> list[str]:
    return [
        "rolling statistics",
        "lag features",
        "rate-of-change features",
        "frequency-domain features",
        "cross-channel interaction features",
    ]
