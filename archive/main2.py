import asyncio
import json
import os
import requests
import numpy as np
import joblib
from scipy.signal import savgol_filter
from datetime import datetime
from learner import run_learning_cycle

# --- CONFIGURATION ---
SIMULATION_MODE = True 
STATE_FILE = "data/bot_state.json"
MODEL_FILE = "data/sl_tp_model.pkl"
SYMBOL = "ETHUSDT"
TIMEFRAME = "4h"

# --- DEFAULTS ---
DEFAULT_SL = 0.03 
DEFAULT_TP = 0.05 

def fetch_candles():
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval={TIMEFRAME}&limit=100"
        data = requests.get(url).json()
        return np.array([float(x[4]) for x in data])
    except:
        return np.array([])

def fetch_smart_money():
    """Fetches the Whale Positions (Top Trader L/S Ratio)"""
    try:
        # Binance Futures API (Public)
        url = f"https://fapi.binance.com/futures/data/topLongShortAccountRatio?symbol={SYMBOL}&period=5m&limit=1"
        data = requests.get(url).json()
        if data:
            return float(data[0]['longShortRatio'])
    except Exception as e:
        print(f"⚠️ Smart Money Data Failed: {e}")
    return 1.0 # Neutral fallback

def calculate_quantum_wave(closes):
    if len(closes) < 21: return 0, 0
    trend = savgol_filter(closes, 21, 3)
    slope = trend[-1] - trend[-2]
    spread = abs(closes[-1] - trend[-1]) / trend[-1]
    return slope, spread

def get_sentiment():
    try:
        fng = requests.get("https://api.alternative.me/fng/?limit=1").json()
        return int(fng['data'][0]['value'])
    except:
        return 50

# --- AI PREDICTION ENGINE ---
def calc_features_for_ai(prices):
    if len(prices) < 50: return None
    smooth = savgol_filter(prices, 21, 3)
    slope = smooth[-1] - smooth[-2]
    volatility = np.std(prices[-20:])
    delta = np.diff(prices)
    gain = delta[delta > 0].mean() if len(delta[delta > 0]) > 0 else 0
    loss = -delta[delta < 0].mean() if len(delta[delta < 0]) > 0 else 1
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return [slope, volatility, rsi]

def get_ai_parameters(closes):
    if not os.path.exists(MODEL_FILE): return DEFAULT_SL, DEFAULT_TP
    try:
        model = joblib.load(MODEL_FILE)
        features = calc_features_for_ai(closes)
        if not features: return DEFAULT_SL, DEFAULT_TP
        prediction = model.predict([features])[0]
        ai_sl = max(0.01, min(prediction[0], 0.10)) 
        ai_tp = max(0.02, min(prediction[1], 0.20)) 
        return ai_sl, ai_tp
    except:
        return DEFAULT_SL, DEFAULT_TP

def update_state(status, is_open, entry, price, fng, slope, spread, win_prob, whale_ratio, decision, reason, suggest_sl, suggest_tp):
    state = {
        "status": status, "is_open": is_open, "entry_price": entry,
        "current_price": price, "last_update": datetime.now().strftime("%H:%M:%S"),
        "sentiment": fng, "slope": slope, "spread": spread, 
        "win_probability": win_prob, "whale_ratio": whale_ratio, # <--- NEW FIELD
        "decision": decision, "reason": reason,
        "suggested_sl": suggest_sl, "suggested_tp": suggest_tp 
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

async def main():
    print(f"🧠 Quantum Advisor v5.1 (Whale Watch) | Mode: MONITORING")
    
    is_in_position = False
    entry_price = 0
    
    current_win_prob = run_learning_cycle() 
    last_learning_time = datetime.now()
    
    while True:
        try:
            if (datetime.now() - last_learning_time).seconds > 14400:
                current_win_prob = run_learning_cycle()
                last_learning_time = datetime.now()

            closes = fetch_candles()
            price = closes[-1] if len(closes) > 0 else 0
            fng = get_sentiment()
            slope, vol = calculate_quantum_wave(closes)
            whale_ratio = fetch_smart_money() # <--- NEW INPUT
            
            ai_sl_pct, ai_tp_pct = get_ai_parameters(closes)
            suggest_long_tp = price * (1 + ai_tp_pct)
            suggest_long_sl = price * (1 - ai_sl_pct)
            
            decision = "SCANNING"
            reason = f"Slope: {slope:.4f} | Whales: {whale_ratio:.2f}"

            if not is_in_position:
                # --- SMART ENTRY LOGIC ---
                # 1. Classic Trend Check
                trend_valid = slope > 0.5
                
                # 2. Whale Divergence Check
                # If Whales are heavily Long (> 1.2) while Sentiment is Fear (< 40), 
                # we treat that as a SUPER signal.
                whale_divergence = (whale_ratio > 1.2) and (fng < 40)
                
                if trend_valid and (fng < 60 or whale_divergence):
                    
                    # 3. AI Gatekeeper
                    # If Whales are Buying, we lower the AI requirement slightly (from 40% to 30%)
                    required_confidence = 30 if whale_divergence else 40
                    
                    if current_win_prob > required_confidence:
                        decision = "🚀 LONG SIGNAL"
                        reason = f"Trend Up + Whales Long ({whale_ratio:.2f})."
                        if whale_divergence:
                            reason = "🚨 WHALE DIVERGENCE DETECTED. Smart Money is buying the Fear."
                        
                        is_in_position = True
                        entry_price = price
                        print(f"🚀 PAPER BUY @ ${price:.2f}")

            else:
                pnl = ((price - entry_price) / entry_price) * 100
                decision = "PAPER HOLD"
                reason = f"Simulated PnL: {pnl:.2f}% | Whale Ratio: {whale_ratio:.2f}"
                
                if price >= suggest_long_tp or price <= suggest_long_sl:
                    decision = "PAPER CLOSE"
                    is_in_position = False
                    entry_price = 0

            update_state("ACTIVE", is_in_position, entry_price, price, fng, slope, vol, current_win_prob, whale_ratio, decision, reason, suggest_long_sl, suggest_long_tp)
            await asyncio.sleep(60)

        except Exception as e:
            print(f"⚠️ Loop Error: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
