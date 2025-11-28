import numpy as np
import pandas as pd
import requests
import os
import joblib
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime

# --- CONFIGURATION ---
SYMBOL = "ETHUSDT"
TIMEFRAME = "4h" # Matched to Main Bot
MEMORY_FILE = "data/market_memory.csv"
MODEL_FILE_WIN = "data/win_prob_model.pkl" # For "Should I Enter?"
MODEL_FILE_TPSL = "data/sl_tp_model.pkl"   # For "Where do I Exit?" (From ai_optimizer)

def fetch_training_data():
    """Fetches 4H candles for training"""
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval={TIMEFRAME}&limit=1000"
        data = requests.get(url).json()
        df = pd.DataFrame(data, columns=['t','o','h','l','c','v','x','y','z','a','b','d'])
        df = df[['t', 'o', 'h', 'l', 'c', 'v']].astype(float)
        return df
    except Exception as e:
        print(f"⚠️ Learner API Error: {e}")
        return pd.DataFrame()

def calc_features(prices, lows):
    if len(prices) < 50: return None
    
    # 1. Quantum Slope
    smooth = savgol_filter(prices, 21, 3)
    slope = smooth[-1] - smooth[-2]
    
    # 2. Volatility
    volatility = np.std(prices[-20:])
    
    # 3. RSI
    delta = np.diff(prices)
    gain = delta[delta > 0].mean() if len(delta[delta > 0]) > 0 else 0
    loss = -delta[delta < 0].mean() if len(delta[delta < 0]) > 0 else 1
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # 4. Support Proximity
    recent_low = np.min(lows[-50:])
    dist = (prices[-1] - recent_low) / prices[-1]

    return [slope, volatility, rsi, dist]

# --- TASK 1: TRAIN WIN PROBABILITY (CLASSIFIER) ---
def train_win_probability(df):
    X, y = [], []
    closes = df['c'].values
    lows = df['l'].values

    # Look back and label trades as Win (1) or Loss (0)
    for i in range(50, len(df) - 10):
        features = calc_features(closes[i-50:i], lows[i-50:i])
        if features:
            # Simple Logic: Did price go up 2% before down 1%?
            entry = closes[i]
            future = closes[i:i+10]
            if np.max(future) > entry * 1.02:
                y.append(1) # Win
            else:
                y.append(0) # Loss/Neutral
            X.append(features)

    if len(X) > 50:
        model = RandomForestClassifier(n_estimators=100, max_depth=10)
        model.fit(X, y)
        joblib.dump(model, MODEL_FILE_WIN)
        
        # Calculate current probability
        current_feats = calc_features(closes[-50:], lows[-50:])
        prob = model.predict_proba([current_feats])[0][1] * 100
        return prob
    return 50.0

# --- TASK 2: TRAIN TP/SL OPTIMIZER (REGRESSOR) ---
# This is the logic moved from ai_optimizer.py
def train_tpsl_optimizer(df):
    X, y = [], []
    closes = df['c'].values
    lows = df['l'].values
    highs = df['h'].values

    for i in range(50, len(df) - 16):
        features = calc_features(closes[i-50:i], lows[i-50:i])
        if not features: continue

        # Simulate optimal outcome for this specific candle
        entry_price = closes[i]
        future_lows = lows[i+1:i+16]
        future_highs = highs[i+1:i+16]

        best_score = -999
        best_sl = 0.02
        best_tp = 0.03

        # Grid Search
        for sl in [0.01, 0.02, 0.03, 0.04, 0.05]:
            for tp in [0.02, 0.04, 0.06, 0.08, 0.10]:
                stop_price = entry_price * (1 - sl)
                target_price = entry_price * (1 + tp)

                hit_stop = np.any(future_lows <= stop_price)
                hit_target = np.any(future_highs >= target_price)

                if hit_target and not hit_stop:
                    score = tp / sl # Risk:Reward Ratio
                    if score > best_score:
                        best_score = score
                        best_sl = sl
                        best_tp = tp
        
        X.append(features)
        y.append([best_sl, best_tp])

    if len(X) > 50:
        model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1)
        model.fit(X, y)
        joblib.dump(model, MODEL_FILE_TPSL)
        print("✅ TP/SL Model Updated")

# --- MAIN EXECUTION ---
def run_learning_cycle():
    """Called by main.py every hour"""
    df = fetch_training_data()
    if df.empty: return 50.0

    # 1. Train the Optimizer (Regression)
    train_tpsl_optimizer(df)

    # 2. Train the Win Probability (Classification)
    current_prob = train_win_probability(df)
    
    return current_prob

if __name__ == "__main__":
    print(f"Manual Run: {run_learning_cycle()}% Win Prob")
