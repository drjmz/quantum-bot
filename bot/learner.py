import numpy as np
import pandas as pd
import requests
from scipy.signal import savgol_filter

def fetch_training_data(symbol="ETHUSDT", days=7):
    # Fetch 7 days of 15m candles (high resolution for training)
    limit = 1000 
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=15m&limit={limit}"
    data = requests.get(url).json()
    closes = np.array([float(x[4]) for x in data])
    return closes

def simulate_strategy(closes, entry_slope_threshold, stop_loss_pct):
    """
    Runs a fast simulation: 
    If we bought when slope > threshold, did we hit TP before SL?
    """
    capital = 1000
    wins = 0
    losses = 0
    
    # Calculate Indicator
    if len(closes) < 21: return 0
    trend = savgol_filter(closes, 21, 3)
    slopes = np.diff(trend) # Change in trend
    
    in_position = False
    entry_price = 0
    
    for i in range(21, len(closes)-1):
        price = closes[i]
        slope = slopes[i-1]
        
        if not in_position:
            # TRY TO BUY
            if slope > entry_slope_threshold:
                in_position = True
                entry_price = price
        else:
            # MANAGE TRADE
            pnl = (price - entry_price) / entry_price
            
            # Target 2x Stop Loss (Standard Risk/Reward)
            if pnl > (stop_loss_pct * 2): 
                wins += 1
                in_position = False
            elif pnl < -stop_loss_pct:
                losses += 1
                in_position = False
                
    total_trades = wins + losses
    if total_trades == 0: return 0
    return (wins / total_trades) * 100  # Return Win Rate %

def run_learning_cycle():
    """
    The 'Shadow Dojo': Tests different sensitivities to find the current win probability.
    """
    try:
        data = fetch_training_data()
        
        # We test 3 'Personalities'
        # 1. Aggressive (Buy on tiny curve)
        win_rate_aggr = simulate_strategy(data, entry_slope_threshold=0.01, stop_loss_pct=0.01)
        
        # 2. Balanced (Buy on solid curve)
        win_rate_bal = simulate_strategy(data, entry_slope_threshold=0.5, stop_loss_pct=0.02)
        
        # 3. Conservative (Buy on strong hook)
        win_rate_safe = simulate_strategy(data, entry_slope_threshold=2.0, stop_loss_pct=0.05)
        
        # Average them to get a "Market Health Score"
        market_score = (win_rate_aggr + win_rate_bal + win_rate_safe) / 3
        
        print(f"🎓 Training Complete. Market Win Probability: {market_score:.1f}%")
        return market_score
        
    except Exception as e:
        print(f"⚠️ Learning Failed: {e}")
        return 50.0 # Default Neutral
