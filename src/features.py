from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag, rolling indicators, and technical features for prediction."""
    data = df.copy()

    # Original Features
    data["return_1d"] = data["close"].pct_change()
    data["return_5d"] = data["close"].pct_change(5)
    data["sma_5"] = data["close"].rolling(5).mean()
    data["sma_10"] = data["close"].rolling(10).mean()
    data["volatility_10"] = data["close"].rolling(10).std()

    # New Features: RSI (14)
    delta = data["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)  # Avoid division by zero
    data["rsi_14"] = 100 - (100 / (1 + rs))

    # New Features: Bollinger Bands (20)
    data["bb_mid"] = data["close"].rolling(window=20).mean()
    data["bb_std"] = data["close"].rolling(window=20).std()
    data["bb_upper"] = data["bb_mid"] + (data["bb_std"] * 2)
    data["bb_lower"] = data["bb_mid"] - (data["bb_std"] * 2)

    # Target: Next day's close
    data["target"] = data["close"].shift(-1)
    return data.dropna()
