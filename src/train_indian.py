from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_loader import load_stock_data
from features import build_features
from model import train_and_save

INDIAN_STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

def main():
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    
    all_frames = []
    
    for symbol in INDIAN_STOCKS:
        print(f"Downloading {symbol}...")
        try:
            dataset = load_stock_data(symbol, start_date, end_date)
            frame = build_features(dataset.frame)
            all_frames.append(frame)
        except Exception as e:
            print(f"Failed to load {symbol}: {e}")
            
    if not all_frames:
        print("No data loaded. Exiting.")
        return
        
    combined_df = pd.concat(all_frames).sort_index()
    print(f"Combined dataset size: {len(combined_df)} rows.")
    
    print("Training specialized Indian Market model...")
    result = train_and_save(combined_df, MODEL_PATH)
    
    print(f"Training Complete!")
    print(f"MAE: {result.mae:.4f}")
    print(f"R2 : {result.r2:.4f}")
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
