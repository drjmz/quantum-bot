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

def get_data():
    print(f"📥 Downloading Market Data ({DAYS} Days)...")
    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval=15m&limit=1000"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=['t','o','h','l','c','v','x','y','z','a','b','d'])
    df['c'] = df['c'].astype(float)
    df['l'] = df['l'].astype(float)
    df['h'] = df['h'].astype(float)
    return df

def calc_slope(prices):
    if len(prices) < 21: return 0
    trend = savgol_filter(prices, 21, 3)
    return trend[-1] - trend[-2]

def run_high_lev_matrix():
    df = get_data()
    prices = df['c'].values
    lows = df['l'].values
    highs = df['h'].values
    
    best_profit = -9999
    best_config = None
    
    print(f"\n🚀 RUNNING HIGH LEVERAGE SIMULATION (1x - 50x)...")
    print(f"{'LEV':<5} {'SL%':<5} {'TP%':<5} {'PROFIT':<10} {'WIN RATE'} {'MAX DD'}")
    print("-" * 55)

    # 1. LOOP LEVERAGE (1x to 50x in steps of 5)
    for lev in [1, 5, 10, 20, 30, 40, 50]:
        
        # Calculate Liquidation Threshold (approx 80% of margin)
        liq_pct = 0.80 / lev 
        
        # 2. LOOP STOP LOSS (Must be safer than liquidation)
        # We test SL from 0.5% up to (Liquidation - 0.5%)
        max_sl = (liq_pct * 100) - 0.5
        if max_sl < 0.5: max_sl = 0.5
        
        sl_range = np.linspace(0.5, max_sl, num=4) # Test 4 safe SL points
        
        for sl_val in sl_range:
            sl = round(sl_val, 2)
            
            # 3. LOOP TAKE PROFIT (1% to 10%)
            for tp in [1, 2, 3, 5, 8, 10]:
                
                balance = CAPITAL
                max_balance = CAPITAL
                drawdown = 0
                wins = 0
                trades = 0
                active = None 
                
                start_index = len(prices) - (DAYS * 24 * 4)
                
                for i in range(start_index, len(prices)):
                    price = prices[i]
                    
                    if active is None:
                        # ENTRY (Trend Hook)
                        prev = prices[i-100:i+1]
                        slope = calc_slope(prev)
                        
                        if slope > 0.5:
                            active = {'entry': price, 'type': 'LONG'}
                        elif slope < -0.5:
                            active = {'entry': price, 'type': 'SHORT'}

                    else:
                        # EXIT (Check Lows/Highs for wicks)
                        entry = active['entry']
                        low = lows[i]
                        high = highs[i]
                        
                        pnl = 0
                        closed = False
                        
                        if active['type'] == 'LONG':
                            # Did we hit SL or Liquidation?
                            if low <= entry * (1 - (sl/100)):
                                pnl = -(COLLATERAL * lev) * (sl/100)
                                closed = True
                            # Did we hit TP?
                            elif high >= entry * (1 + (tp/100)):
                                pnl = (COLLATERAL * lev) * (tp/100)
                                wins += 1
                                closed = True
                        else: # SHORT
                            if high >= entry * (1 + (sl/100)):
                                pnl = -(COLLATERAL * lev) * (sl/100)
                                closed = True
                            elif low <= entry * (1 - (tp/100)):
                                pnl = (COLLATERAL * lev) * (tp/100)
                                wins += 1
                                closed = True
                        
                        if closed:
                            balance += pnl
                            if balance > max_balance: max_balance = balance
                            dd = (max_balance - balance) / max_balance * 100
                            if dd > drawdown: drawdown = dd
                            
                            active = None
                            trades += 1
                            
                            # Bust Check
                            if balance < 10: break 
                
                net = balance - CAPITAL
                
                # Filter: Profitable AND Drawdown < 30% (Safety first)
                if net > 0 and drawdown < 30:
                    wr = (wins/trades)*100 if trades > 0 else 0
                    print(f"{lev}x{'':<3} {sl}%{'':<3} {tp}%{'':<3} ${net:<9.2f} {wr:.0f}%{'':<6} {drawdown:.1f}%")
                    
                    if net > best_profit:
                        best_profit = net
                        best_config = (lev, sl, tp)

    print("-" * 55)
    if best_config:
        print(f"🏆 HIGH LEVERAGE CHAMPION")
        print(f"   Leverage:    {best_config[0]}x")
        print(f"   Stop Loss:   {best_config[1]}%")
        print(f"   Take Profit: {best_config[2]}%")
        print(f"   Net Profit:  ${best_profit:.2f}")
    else:
        print("⚠️ No safe high-leverage settings found.")

if __name__ == "__main__":
    run_high_lev_matrix()
