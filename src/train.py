from __future__ import annotations

import argparse

from data_loader import load_stock_data
from features import build_features
from model import train_and_save


def parse_args():
    parser = argparse.ArgumentParser(description="Train stock price prediction model.")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-01-01", help="End date YYYY-MM-DD")
    parser.add_argument("--model-path", default="models/model.joblib", help="Model output path")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_stock_data(args.symbol, args.start, args.end)
    frame = build_features(dataset.frame)
    metrics = train_and_save(frame, args.model_path)

    print(f"Symbol: {dataset.symbol}")
    print(f"Rows used: {len(frame)}")
    print(f"MAE: {metrics.mae:.4f}")
    print(f"R2: {metrics.r2:.4f}")
    print(f"Saved model to: {args.model_path}")


if __name__ == "__main__":
    main()
