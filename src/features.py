from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag and rolling indicators for next-day close prediction."""
    data = df.copy()
    data["return_1d"] = data["close"].pct_change()
    data["return_5d"] = data["close"].pct_change(5)
    data["sma_5"] = data["close"].rolling(5).mean()
    data["sma_10"] = data["close"].rolling(10).mean()
    data["volatility_10"] = data["close"].rolling(10).std()
    data["target"] = data["close"].shift(-1)
    return data.dropna()
