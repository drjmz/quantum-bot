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
    # Get Price Data
    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval=15m&limit=1000"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=['t','o','h','l','c','v','x','y','z','a','b','d'])
    df['c'] = df['c'].astype(float)
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    return df

def get_fng():
    # Get Fear Data Map
    try:
        url = "https://api.alternative.me/fng/?limit=30"
        data = requests.get(url).json()['data']
        fng_map = {}
        for day in data:
            date_str = datetime.fromtimestamp(int(day['timestamp'])).strftime('%Y-%m-%d')
            fng_map[date_str] = int(day['value'])
        return fng_map
    except:
        return {}

def calc_slope(prices):
    if len(prices) < 21: return 0
    trend = savgol_filter(prices, 21, 3)
    return trend[-1] - trend[-3]

def run_matrix():
    print("📥 Downloading Data & Sentiment...")
    df = get_data()
    prices = df['c'].values
    times = df['t']
    fng_map = get_fng()
    
    print(f"\n🧩 CALCULATING PROFIT MATRIX (7 Days w/ Fear Filter)...")
    print(f"{'SL%':<5} {'TP%':<5} {'NET PROFIT':<12} {'WIN RATE'}")
    print("-" * 35)

    best_profit = -9999
    best_config = (0, 0)

    # LOOP SL (2-6%) and TP (2-15%)
    # Adjusted range slightly based on previous findings
    for sl in range(2, 7):       
        for tp in range(2, 16):  
            
            balance = CAPITAL
            wins = 0
            trades = 0
            active = None 
            
            start_index = len(prices) - (DAYS * 24 * 4)
            
            for i in range(start_index, len(prices)):
                price = prices[i]
                date_str = times.iloc[i].strftime('%Y-%m-%d')
                fng = fng_map.get(date_str, 50)
                
                if active is None:
                    # ENTRY: Slope > 0 AND Fear < 40
                    if fng < 40:
                        # FIX IS HERE: Include current candle (i+1)
                        prev = prices[i-50:i+1] 
                        if calc_slope(prev) > 0:
                             active = {'entry': price}
                else:
                    # EXIT
                    entry = active['entry']
                    pnl = (price - entry) / entry
                    
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
            
            # Only print Profitable ones
            if net > 0:
                wr = (wins/trades)*100
                print(f"{sl}%   {tp}%    ${net:<11.2f} {wr:.0f}%")
                
                if net > best_profit:
                    best_profit = net
                    best_config = (sl, tp)

    print("-" * 35)
    if best_profit > -9999:
        print(f"🏆 CHAMPION: SL {best_config[0]}% | TP {best_config[1]}%")
        print(f"💰 MAX PROFIT: ${best_profit:.2f}")
    else:
        print("⚠️ No profitable settings found. Market is too choppy.")

if __name__ == "__main__":
    run_matrix()
