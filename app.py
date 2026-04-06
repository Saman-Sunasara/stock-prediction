from __future__ import annotations

import sys
from pathlib import Path
from datetime import date

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_loader import load_stock_data  # noqa: E402
from features import build_features  # noqa: E402
from model import FEATURE_COLUMNS, load_model, train_and_save  # noqa: E402

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"


def run_training(symbol: str, start_date: str, end_date: str) -> str:
    dataset = load_stock_data(symbol, start_date, end_date)
    frame = build_features(dataset.frame)
    metrics = train_and_save(frame, DEFAULT_MODEL_PATH)
    return (
        f"Trained on {len(frame)} rows for {dataset.symbol}. "
        f"MAE={metrics.mae:.4f}, R2={metrics.r2:.4f}"
    )


def run_prediction(symbol: str, start_date: str, end_date: str):
    model = load_model(DEFAULT_MODEL_PATH)
    dataset = load_stock_data(symbol, start_date, end_date)
    frame = build_features(dataset.frame)

    latest = frame[FEATURE_COLUMNS].iloc[[-1]]
    pred = float(model.predict(latest)[0])
    last_close = float(frame["close"].iloc[-1])
    delta = pred - last_close
    delta_pct = (delta / last_close) * 100
    return dataset.symbol, last_close, pred, delta, delta_pct


def main():
    st.set_page_config(page_title="Stock Predictor", page_icon=":chart_with_upwards_trend:")
    st.title("Stock Prediction App")
    st.caption("Educational next-day close prediction with RandomForest.")

    with st.sidebar:
        st.header("Inputs")
        symbol = st.text_input("Ticker symbol", value="AAPL").upper().strip()
        train_start = st.date_input("Train start date", value=date(2020, 1, 1))
        train_end = st.date_input("Train end date", value=date(2026, 1, 1))
        pred_start = st.date_input("Predict window start date", value=date(2024, 1, 1))
        pred_end = st.date_input("Predict window end date", value=date(2026, 1, 1))

    train_start_s = str(train_start)
    train_end_s = str(train_end)
    pred_start_s = str(pred_start)
    pred_end_s = str(pred_end)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train model", use_container_width=True):
            try:
                with st.spinner("Training model..."):
                    message = run_training(symbol, train_start_s, train_end_s)
                st.success(message)
                st.info(f"Saved to `{DEFAULT_MODEL_PATH}`")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    with col2:
        if st.button("Predict next close", use_container_width=True):
            if not DEFAULT_MODEL_PATH.exists():
                st.warning("Train the model first.")
            else:
                try:
                    with st.spinner("Running prediction..."):
                        ticker, last_close, pred, delta, delta_pct = run_prediction(
                            symbol, pred_start_s, pred_end_s
                        )
                    st.subheader(f"Prediction for {ticker}")
                    st.metric("Last known close", f"{last_close:.2f}")
                    st.metric("Predicted next close", f"{pred:.2f}", f"{delta:+.2f} ({delta_pct:+.2f}%)")
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

    st.markdown("---")
    st.write("This app is for educational purposes only and not financial advice.")


if __name__ == "__main__":
    main()
