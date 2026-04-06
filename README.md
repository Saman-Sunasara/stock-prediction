# Stock Prediction Project

A simple machine learning pipeline that downloads historical stock data, trains a model to predict the next day's closing price, and runs predictions from the command line.

## Features

- Downloads OHLCV stock data from Yahoo Finance
- Builds technical indicators (returns, moving averages, volatility)
- Trains a `RandomForestRegressor`
- Saves and reuses trained model for inference

## Project Structure

```text
stock-prediction/
  src/
    data_loader.py
    features.py
    model.py
    train.py
    predict.py
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

## Train

```bash
python src/train.py --symbol AAPL --start 2020-01-01 --end 2026-01-01
```

## Predict

```bash
python src/predict.py --symbol AAPL --start 2024-01-01 --end 2026-01-01
```

## Notes

- This is an educational project and not financial advice.
- For production-quality forecasting, add richer features, cross-validation, and model monitoring.
