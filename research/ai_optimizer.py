import numpy as np
import pandas as pd
import requests
import os
import joblib
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# --- CONFIG ---
SYMBOL = "ETHUSDT"
FETCH_DAYS = 30
MEMORY_FILE = "data/market_memory.csv"
MODEL_FILE = "data/sl_tp_model.pkl"

def fetch_fresh_data():
    print(f"📥 Fetching latest {FETCH_DAYS} days from Binance...")
    limit = 1000
    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval=15m&limit={limit}"
    try:
        data = requests.get(url).json()
        df = pd.DataFrame(data, columns=['t','o','h','l','c','v','x','y','z','a','b','d'])
        df = df[['t', 'o', 'h', 'l', 'c', 'v']].astype(float)
        return df
    except Exception as e:
        print(f"⚠️ API Error: {e}")
        return pd.DataFrame()

def manage_memory():
    new_df = fetch_fresh_data()
    if new_df.empty: return pd.DataFrame()
    
    if os.path.exists(MEMORY_FILE):
        print("📂 Loading existing Market Memory...")
        old_df = pd.read_csv(MEMORY_FILE)
        combined = pd.concat([old_df, new_df]).drop_duplicates(subset=['t'], keep='last').sort_values('t')
        combined.to_csv(MEMORY_FILE, index=False)
        return combined
    else:
        print("✨ Creating new Market Memory Vault...")
        new_df.to_csv(MEMORY_FILE, index=False)
        return new_df

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
    
    # 4. NEW: Support Proximity (Simulated for training)
    # In live trading we check the Order Book. Here, we check local lows.
    recent_low = np.min(lows[-50:])
    dist_to_support = (prices[-1] - recent_low) / prices[-1]
    
    return [slope, volatility, rsi, dist_to_support]

def find_optimal_params(entry_idx, df):
    entry_price = df.iloc[entry_idx]['c']
    future_window = df.iloc[entry_idx+1 : entry_idx+16] 
    
    best_score = -999
    best_sl = 0.03
    best_tp = 0.05
    
    for sl in [0.02, 0.03, 0.04, 0.05, 0.06]:
        for tp in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
            stop_price = entry_price * (1 - sl)
            target_price = entry_price * (1 + tp)
            lows = future_window['l'].values
            highs = future_window['h'].values
            
            hit_stop = np.any(lows <= stop_price)
            hit_target = np.any(highs >= target_price)
            
            if hit_target and not hit_stop:
                score = tp / sl
                if score > best_score:
                    best_score = score
                    best_sl = sl
                    best_tp = tp
    return best_sl, best_tp

def train_model():
    df = manage_memory()
    if df.empty or len(df) < 200: return

    X = []
    y = []
    
    print(f"🎓 Training AI on {len(df)} historical candles...")
    closes = df['c'].values
    lows = df['l'].values
    
    for i in range(50, len(df) - 20):
        prices = closes[i-50:i]
        local_lows = lows[i-50:i]
        features = calc_features(prices, local_lows)
        
        if features:
            opt_sl, opt_tp = find_optimal_params(i, df)
            X.append(features)
            y.append([opt_sl, opt_tp])
            
    if len(X) > 0:
        model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1)
        model.fit(X, y)
        joblib.dump(model, MODEL_FILE)
        print(f"✅ Model Updated & Saved. Features: Slope, Volatility, RSI, SupportDist")
    else:
        print("⚠️ Extraction failed.")

if __name__ == "__main__":
    train_model()
