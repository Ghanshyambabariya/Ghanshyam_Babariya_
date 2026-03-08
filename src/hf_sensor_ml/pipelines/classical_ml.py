"""Baseline machine-learning models for engineered sensor features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def train_baseline_models(features: pd.DataFrame, *, label_column: str, random_state: int = 42) -> dict:
    if label_column not in features.columns:
        raise ValueError(f"Feature dataset must contain label column '{label_column}'")

    X = features.drop(columns=[label_column])
    y = features[label_column]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.25,
        random_state=random_state,
        stratify=y_encoded,
    )

    numeric_columns = list(X.columns)
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_columns)],
        remainder="drop",
    )

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=4000, multi_class="auto")),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", "passthrough"),
                ("model", RandomForestClassifier(n_estimators=300, random_state=random_state)),
            ]
        ),
    }

    results: dict[str, object] = {
        "label_mapping": {int(i): label for i, label in enumerate(label_encoder.classes_)},
        "models": {},
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results["models"][name] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
            "classification_report": classification_report(
                y_test,
                predictions,
                target_names=label_encoder.classes_,
                zero_division=0,
                output_dict=True,
            ),
        }
        if name == "random_forest":
            fitted_model = model.named_steps["model"]
            importances = fitted_model.feature_importances_
            order = np.argsort(importances)[::-1][:10]
            results["models"][name]["top_features"] = [
                {"feature": numeric_columns[idx], "importance": round(float(importances[idx]), 4)}
                for idx in order
            ]

    return results
