"""Sequence models for windowed sensor classification."""

from __future__ import annotations

import copy
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_sequence_dataset(
    df: pd.DataFrame,
    *,
    sample_rate_hz: int,
    sensor_columns: list[str],
    label_column: str,
    window_seconds: float,
    stride_seconds: float,
    downsample_factor: int,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    if label_column not in df.columns:
        raise ValueError(f"Raw dataset must contain label column '{label_column}' for sequence training")

    window_size = int(window_seconds * sample_rate_hz)
    stride = int(stride_seconds * sample_rate_hz)
    sequences: list[np.ndarray] = []
    labels: list[str] = []

    for start in range(0, len(df) - window_size + 1, stride):
        window = df.iloc[start : start + window_size]
        sequence = window[sensor_columns].to_numpy(dtype=np.float32)[::downsample_factor]
        sequences.append(sequence)
        labels.append(str(window[label_column].mode().iloc[0]))

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return np.stack(sequences), encoded_labels.astype(np.int64), encoder


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(inputs)
        hidden = self.dropout(hidden[-1])
        return self.output(hidden)


class TemporalCNNClassifier(nn.Module):
    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.output = nn.Linear(64, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.transpose(1, 2)
        x = self.network(x).squeeze(-1)
        return self.output(x)


def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_tensor: torch.Tensor,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    epochs: int,
    learning_rate: float,
) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(test_tensor).argmax(dim=1).cpu().numpy()
        current_f1 = f1_score(y_test, predictions, average="macro")
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_predictions = model(test_tensor).argmax(dim=1).cpu().numpy()

    return {
        "accuracy": round(float(accuracy_score(y_test, final_predictions)), 4),
        "macro_f1": round(float(f1_score(y_test, final_predictions, average="macro")), 4),
        "classification_report": classification_report(
            y_test,
            final_predictions,
            target_names=label_encoder.classes_,
            zero_division=0,
            output_dict=True,
        ),
    }


def train_sequence_models(
    df: pd.DataFrame,
    *,
    sample_rate_hz: int,
    sensor_columns: list[str],
    label_column: str,
    window_seconds: float,
    stride_seconds: float,
    downsample_factor: int,
    random_state: int = 42,
) -> dict:
    _set_seed(random_state)
    sequences, labels, encoder = make_sequence_dataset(
        df,
        sample_rate_hz=sample_rate_hz,
        sensor_columns=sensor_columns,
        label_column=label_column,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        downsample_factor=downsample_factor,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        sequences,
        labels,
        test_size=0.25,
        random_state=random_state,
        stratify=labels,
    )

    train_mean = X_train.mean(axis=(0, 1), keepdims=True)
    train_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)

    lstm = LSTMClassifier(input_size=X_train.shape[2], hidden_size=32, num_classes=len(encoder.classes_))
    temporal_cnn = TemporalCNNClassifier(input_channels=X_train.shape[2], num_classes=len(encoder.classes_))

    return {
        "label_mapping": {int(i): label for i, label in enumerate(encoder.classes_)},
        "models": {
            "lstm": _train_model(
                lstm,
                train_loader,
                test_tensor,
                y_test,
                encoder,
                epochs=10,
                learning_rate=0.001,
            ),
            "temporal_cnn": _train_model(
                temporal_cnn,
                train_loader,
                test_tensor,
                y_test,
                encoder,
                epochs=10,
                learning_rate=0.001,
            ),
        },
    }
