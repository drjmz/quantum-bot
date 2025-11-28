import pandas as pd
import numpy as np
import requests
from scipy.signal import savgol_filter
from datetime import datetime

# --- CONFIG ---
SYMBOL = "ETHUSDT"
DAYS = 7
CAPITAL = 100.0
COLLATERAL = 10.0
LEVERAGE = 2.0

def get_data():
    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval=15m&limit=1000"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=['t','o','h','l','c','v','x','y','z','a','b','d'])
    df['c'] = df['c'].astype(float)
    return df['c'].values

def calc_slope(prices):
    if len(prices) < 21: return 0
    trend = savgol_filter(prices, 21, 3)
    # We look at the immediate momentum
    return trend[-1] - trend[-2]

def run_matrix():
    print("📥 Downloading Price Data (Ignoring Sentiment)...")
    prices = get_data()
    
    print(f"\n🧩 CALCULATING UNCHAINED MATRIX (Pure Trend)...")
    print(f"{'SL%':<5} {'TP%':<5} {'NET PROFIT':<12} {'WIN RATE'}")
    print("-" * 35)

    best_profit = -9999
    best_config = (0, 0)

    # LOOP SL (1-6%) and TP (1-15%)
    for sl in range(1, 7):       
        for tp in range(1, 16):  
            
            balance = CAPITAL
            wins = 0
            trades = 0
            active = None 
            
            start_index = len(prices) - (DAYS * 24 * 4)
            
            for i in range(start_index, len(prices)):
                price = prices[i]
                
                # LOOKBACK 100
                prev = prices[i-100:i+1]
                slope = calc_slope(prev)

                if active is None:
                    # PURE MATH ENTRY
                    # If slope is significantly positive/negative
                    if slope > 0.5:
                        active = {'entry': price, 'type': 'LONG'}
                    elif slope < -0.5:
                        active = {'entry': price, 'type': 'SHORT'}

                else:
                    # EXIT LOGIC
                    entry = active['entry']
                    if active['type'] == 'LONG':
                        pnl = (price - entry) / entry
                    else: # SHORT
                        pnl = (entry - price) / entry
                    
                    if pnl >= (tp/100.0): 
                        balance += (COLLATERAL * LEVERAGE) * (tp/100.0)
                        active = None
                        wins += 1
                        trades += 1
                    elif pnl <= -(sl/100.0): 
                        balance -= (COLLATERAL * LEVERAGE) * (sl/100.0)
                        active = None
                        trades += 1
            
            net = balance - CAPITAL
            
            if net > 0:
                wr = (wins/trades)*100 if trades > 0 else 0
                print(f"{sl}%   {tp}%    ${net:<11.2f} {wr:.0f}%")
                if net > best_profit:
                    best_profit = net
                    best_config = (sl, tp)

    print("-" * 35)
    if best_profit > -9999:
        print(f"🏆 CHAMPION: SL {best_config[0]}% | TP {best_config[1]}%")
        print(f"💰 MAX PROFIT: ${best_profit:.2f}")

if __name__ == "__main__":
    run_matrix()
