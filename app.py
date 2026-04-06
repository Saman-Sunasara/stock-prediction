from __future__ import annotations

import sys
from pathlib import Path
from datetime import date

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_loader import load_stock_data  # noqa: E402
from features import build_features  # noqa: E402
from model import FEATURE_COLUMNS, load_model, train_and_save, TrainResult  # noqa: E402

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"


def plot_stock_and_indicators(df: pd.DataFrame, symbol: str):
    """Create interactive plotly chart with price, BB, and RSI."""
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1, 
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{symbol} Price & Bollinger Bands", "RSI (14)")
    )

    # Candlestick / Line for Price
    fig.add_trace(
        go.Scatter(x=df.index, y=df["close"], name="Close", line=dict(color="blue")),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper", line=dict(color="rgba(173, 216, 230, 0.5)", dash="dash")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower", line=dict(color="rgba(173, 216, 230, 0.5)", dash="dash"), fill="tonexty"),
        row=1, col=1
    )

    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df["rsi_14"], name="RSI", line=dict(color="purple")),
        row=2, col=1
    )
    # RSI thresholds
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(height=600, template="plotly_dark", showlegend=True)
    return fig


def plot_performance(y_test: pd.Series, preds: pd.Series):
    """Create actual vs predicted chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test, name="Actual", line=dict(color="cyan")))
    fig.add_trace(go.Scatter(x=y_test.index, y=preds, name="Predicted", line=dict(color="magenta", dash="dot")))
    
    fig.update_layout(
        title="Model Performance: Actual vs Predicted (Test Set)",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=400
    )
    return fig


def main():
    st.set_page_config(page_title="Stock Predictor Pro", page_icon=":chart_with_upwards_trend:", layout="wide")
    st.title("🚀 Stock Prediction Dashboard")
    st.markdown("Enhancing financial forecasting with technical indicators and ensemble learning.")

    with st.sidebar:
        st.header("⚙️ Configuration")
        symbol = st.text_input("Ticker Symbol", value="AAPL").upper().strip()
        st.divider()
        st.subheader("🗓️ Data Range")
        train_start = st.date_input("Train Start", value=date(2020, 1, 1))
        train_end = st.date_input("Train End", value=date(2024, 1, 1))
        pred_start = st.date_input("Prediction Start", value=date(2024, 1, 1))
        pred_end = st.date_input("Prediction End", value=date(2026, 1, 1))
        
        st.divider()
        if st.button("🚀 Train New Model", use_container_width=True):
            try:
                with st.spinner(f"Fetching data for {symbol}..."):
                    dataset = load_stock_data(symbol, str(train_start), str(train_end))
                    frame = build_features(dataset.frame)
                
                with st.spinner("Training Random Forest Ensembles..."):
                    result = train_and_save(frame, DEFAULT_MODEL_PATH)
                    st.session_state["train_result"] = result
                st.success(f"Model trained! MAE: {result.mae:.4f}, R2: {result.r2:.4f}")
            except Exception as e:
                st.error(f"Training Error: {e}")

    # Tabs for main content
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Technical Analysis", "🧠 Model Performance"])

    with tab1:
        col1, col2 = st.columns([1, 1])
        
        # We need data to show overview if model exists
        if DEFAULT_MODEL_PATH.exists():
            try:
                dataset = load_stock_data(symbol, str(pred_start), str(pred_end))
                frame = build_features(dataset.frame)
                
                model = load_model(DEFAULT_MODEL_PATH)
                latest = frame[FEATURE_COLUMNS].iloc[[-1]]
                pred_val = float(model.predict(latest)[0])
                last_close = float(frame["close"].iloc[-1])
                
                delta = pred_val - last_close
                delta_pct = (delta / last_close) * 100
                
                with col1:
                    st.subheader(f"Next-Day Prediction: {symbol}")
                    st.metric("Last Close", f"${last_close:.2f}")
                    st.metric("Predicted Close", f"${pred_val:.2f}", f"{delta:+.2f} ({delta_pct:+.2f}%)")
                    
                with col2:
                    st.subheader("Project Summary")
                    st.info("This model uses a Random Forest Regressor trained on historical OHLCV data and technical indicators (Returns, SMA, Volatility, RSI, BB).")
                    
            except Exception as e:
                st.warning(f"Could not run prediction: {e}")
        else:
            st.warning("Please train the model from the sidebar first.")

    with tab2:
        if DEFAULT_MODEL_PATH.exists():
             try:
                dataset = load_stock_data(symbol, str(pred_start), str(pred_end))
                frame = build_features(dataset.frame)
                st.plotly_chart(plot_stock_and_indicators(frame, symbol), use_container_width=True)
             except Exception as e:
                 st.error(f"Error drawing charts: {e}")
        else:
            st.info("Train the model to see technical indicators.")

    with tab3:
        if "train_result" in st.session_state:
            res: TrainResult = st.session_state["train_result"]
            st.subheader("Training Metrics")
            m1, m2 = st.columns(2)
            m1.metric("Mean Absolute Error (MAE)", f"{res.mae:.4f}")
            m2.metric("R-Squared (R2) Score", f"{res.r2:.2f}")
            
            st.plotly_chart(plot_performance(res.y_test, res.preds), use_container_width=True)
        else:
            st.info("Train the model in this session to see evaluation details.")

    st.divider()
    st.caption("Disclaimer: Not financial advice. For educational purposes only.")


if __name__ == "__main__":
    main()
