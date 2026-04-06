from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import yfinance as yf


@dataclass
class StockDataset:
    symbol: str
    frame: pd.DataFrame


def load_stock_data(symbol: str, start_date: str, end_date: str) -> StockDataset:
    """Download OHLCV data from Yahoo Finance."""
    _ = datetime.fromisoformat(start_date)
    _ = datetime.fromisoformat(end_date)

    frame = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if frame.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'.")

    frame = frame.rename(columns=str.lower)
    if "close" not in frame.columns:
        raise ValueError("Downloaded data does not include close prices.")

    return StockDataset(symbol=symbol.upper(), frame=frame)
