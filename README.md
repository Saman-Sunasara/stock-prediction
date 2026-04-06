# Indian Stock Prediction Dashboard 🇮🇳

🚀 **Live App**: [https://stock-prediction-jyzgafotcusffaavvk5lqb.streamlit.app/](https://stock-prediction-jyzgafotcusffaavvk5lqb.streamlit.app/)

A specialized machine learning pipeline for the **Indian Stock Market (NSE)**. It downloads historical data for major NIFTY 50 stocks, trains a specialized RandomForest model, and provides an interactive Plotly dashboard.

## Features

- 🇮🇳 Specialized for **National Stock Exchange (NSE)** equities.
- 📊 Interactive **Plotly Dashboard** with RSI and Bollinger Bands.
- 💰 Default currency set to **₹ (INR)**.
- 🧠 Pre-trained on a combined dataset of NIFTY 50 leaders (RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK).

## Project Structure

```text
stock-prediction/
  src/
    data_loader.py
    features.py
    model.py
    train.py
    train_indian.py  <-- Specialized NSE Training
    predict.py
  app.py
  requirements.txt
  .gitignore
  README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train (Indian Market)

To retrain the model on the top 5 NIFTY 50 stocks:
```bash
python src/train_indian.py
```

## Run Locally

```bash
streamlit run app.py
```

## GitHub Automation

This repo includes GitHub Actions for automated CI/CD:

- `CI`: Validates code on every push.
- `Container Publish`: Builds and publishes Docker image to **GHCR** (`ghcr.io`).
- `Deploy Hooks`: Triggers deployments on **Render/Railway/Koyeb**.

Expected image: `ghcr.io/saman-sunasara/stock-prediction:latest`

## Notes

- This is an educational project and not financial advice.
- For production-quality forecasting, add richer features and cross-validation.
