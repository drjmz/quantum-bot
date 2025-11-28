import pandas as pd
import numpy as np
import requests
from scipy.signal import savgol_filter
from datetime import datetime

# --- CONFIG ---
SYMBOL = "ETHUSDT"
DAYS = 7
CAPITAL = 100.0
COLLATERAL = 10.0  # Amount of own money per trade

def get_data():
    print(f"📥 Downloading Market Data ({DAYS} Days)...")
    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval=15m&limit=1000"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=['t','o','h','l','c','v','x','y','z','a','b','d'])
    df['c'] = df['c'].astype(float)
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    return df

def calc_slope(prices):
    if len(prices) < 21: return 0
    trend = savgol_filter(prices, 21, 3)
    return trend[-1] - trend[-2]

def run_3d_matrix():
    df = get_data()
    prices = df['c'].values
    
    best_profit = -9999
    best_config = None
    
    print(f"\n🚀 RUNNING 3D SIMULATION (Leverage 2x - 5x)...")
    print(f"{'LEV':<5} {'SL%':<5} {'TP%':<5} {'NET PROFIT':<12} {'WIN RATE'}")
    print("-" * 45)

    # 1. LOOP LEVERAGE
    for lev in [2, 3, 4, 5]:
        
        # 2. LOOP STOP LOSS
        for sl in range(2, 7):       
            
            # 3. LOOP TAKE PROFIT
            for tp in range(2, 16):  
                
                balance = CAPITAL
                wins = 0
                trades = 0
                active = None 
                
                start_index = len(prices) - (DAYS * 24 * 4)
                
                for i in range(start_index, len(prices)):
                    price = prices[i]
                    
                    if active is None:
                        # ENTRY (Pure Trend Logic)
                        prev = prices[i-100:i+1]
                        slope = calc_slope(prev)
                        
                        if slope > 0.5:
                            active = {'entry': price, 'type': 'LONG'}
                        elif slope < -0.5:
                            active = {'entry': price, 'type': 'SHORT'}

                    else:
                        # EXIT
                        entry = active['entry']
                        
                        if active['type'] == 'LONG':
                            pnl_pct = (price - entry) / entry
                        else:
                            pnl_pct = (entry - price) / entry
                        
                        # LEVERAGE MATH
                        # PnL = (Collateral * Leverage) * %Move
                        
                        if pnl_pct >= (tp/100.0): 
                            profit = (COLLATERAL * lev) * (tp/100.0)
                            balance += profit
                            active = None
                            wins += 1
                            trades += 1
                        elif pnl_pct <= -(sl/100.0): 
                            loss = (COLLATERAL * lev) * (sl/100.0)
                            balance -= loss
                            active = None
                            trades += 1
                
                net = balance - CAPITAL
                
                # Filter for only "Great" results to keep log clean
                # Must be profitable AND have > 50% win rate
                if net > 0 and (wins/trades) > 0.5:
                    wr = (wins/trades)*100
                    print(f"{lev}x{'':<3} {sl}%{'':<3} {tp}%{'':<3} ${net:<11.2f} {wr:.0f}%")
                    
                    if net > best_profit:
                        best_profit = net
                        best_config = (lev, sl, tp)

    print("-" * 45)
    if best_config:
        print(f"🏆 GRAND CHAMPION CONFIGURATION")
        print(f"   Leverage:    {best_config[0]}x")
        print(f"   Stop Loss:   {best_config[1]}%")
        print(f"   Take Profit: {best_config[2]}%")
        print(f"   Total Profit: ${best_profit:.2f}")
    else:
        print("⚠️ No safe profitable configurations found.")

if __name__ == "__main__":
    run_3d_matrix()
