from __future__ import annotations

import argparse

from data_loader import load_stock_data
from features import build_features
from model import FEATURE_COLUMNS, load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Predict next-day close price.")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-01-01", help="End date YYYY-MM-DD")
    parser.add_argument("--model-path", default="models/model.joblib", help="Trained model path")
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.model_path)
    dataset = load_stock_data(args.symbol, args.start, args.end)
    frame = build_features(dataset.frame)

    latest = frame[FEATURE_COLUMNS].iloc[[-1]]
    pred = float(model.predict(latest)[0])
    last_close = float(frame["close"].iloc[-1])
    delta = pred - last_close
    delta_pct = (delta / last_close) * 100

    print(f"Symbol: {dataset.symbol}")
    print(f"Last known close: {last_close:.2f}")
    print(f"Predicted next close: {pred:.2f}")
    print(f"Change: {delta:+.2f} ({delta_pct:+.2f}%)")


if __name__ == "__main__":
    main()
