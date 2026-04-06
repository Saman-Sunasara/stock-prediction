# Stock Prediction Project

A simple machine learning pipeline that downloads historical stock data, trains a model to predict the next day's closing price, and runs predictions from the command line.

## Features

- Downloads OHLCV stock data from Yahoo Finance
- Builds technical indicators (returns, moving averages, volatility)
- Trains a `RandomForestRegressor`
- Saves and reuses trained model for inference
- Includes a Streamlit web app for interactive training and prediction

## Project Structure

```text
stock-prediction/
  src/
    data_loader.py
    features.py
    model.py
    train.py
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

## Train

```bash
python src/train.py --symbol AAPL --start 2020-01-01 --end 2026-01-01
```

## Predict

```bash
python src/predict.py --symbol AAPL --start 2024-01-01 --end 2026-01-01
```

## Run Web App (Streamlit)

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [https://share.streamlit.io/](https://share.streamlit.io/).
3. Click **New app**.
4. Select your repository: `Saman-Sunasara/stock-prediction`.
5. Set main file path to `app.py`.
6. Click **Deploy**.

After deployment, Streamlit gives you a public app URL.

## GitHub Automation Added

This repo now includes GitHub Actions:

- `CI` workflow: installs dependencies and validates Python modules on every push/PR.
- `Container Publish` workflow: builds and publishes image to GitHub Container Registry (`ghcr.io`) on every push to `main`.
- `Deploy Hooks` workflow: triggers deployment hooks for Render/Railway/Koyeb on every push to `main` (if secrets are set).

Expected image name:

```text
ghcr.io/saman-sunasara/stock-prediction:latest
```

You can run that image anywhere that supports Docker:

```bash
docker run -p 8501:8501 ghcr.io/saman-sunasara/stock-prediction:latest
```

## Full Auto-Deploy Setup (one-time secrets)

To enable deployment "everywhere", set these **GitHub repository secrets**:

- `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` (optional): also publish image to Docker Hub.
- `RENDER_DEPLOY_HOOK_URL` (optional): trigger Render deploy on push.
- `RAILWAY_DEPLOY_HOOK_URL` (optional): trigger Railway deploy on push.
- `KOYEB_DEPLOY_HOOK_URL` (optional): trigger Koyeb deploy on push.

Path: `GitHub repo -> Settings -> Secrets and variables -> Actions -> New repository secret`

Once these are added, every push to `main` auto-runs:

- CI checks
- image build and publish (GHCR + optional Docker Hub)
- host redeploy webhooks (optional)

## Notes

- This is an educational project and not financial advice.
- For production-quality forecasting, add richer features, cross-validation, and model monitoring.
