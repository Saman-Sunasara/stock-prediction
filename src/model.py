from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

FEATURE_COLUMNS = [
    "close",
    "volume",
    "return_1d",
    "return_5d",
    "sma_5",
    "sma_10",
    "volatility_10",
    "rsi_14",
    "bb_mid",
    "bb_upper",
    "bb_lower",
]


@dataclass
class TrainResult:
    mae: float
    r2: float
    y_test: pd.Series
    preds: pd.Series


def train_and_save(df: pd.DataFrame, model_path: str | Path) -> TrainResult:
    X = df[FEATURE_COLUMNS]
    y = df["target"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds_series = pd.Series(preds, index=y_test.index, name="predicted")

    result = TrainResult(
        mae=float(mean_absolute_error(y_test, preds)),
        r2=float(r2_score(y_test, preds)),
        y_test=y_test,
        preds=preds_series,
    )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return result


def load_model(model_path: str | Path):
    return joblib.load(model_path)
