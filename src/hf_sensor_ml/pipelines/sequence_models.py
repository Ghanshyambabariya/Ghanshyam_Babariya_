"""Advanced sequence-model planning for sensor time series."""


def sequence_model_candidates() -> list[str]:
    return [
        "stacked LSTM",
        "bidirectional LSTM for offline experiments",
        "temporal convolution network",
        "transformer-based time-series encoder",
    ]
