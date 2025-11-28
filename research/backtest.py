import pandas as pd
import numpy as np
import requests
from scipy.signal import savgol_filter
from datetime import datetime

# --- CONFIGURATION ---
SYMBOL = "ETHUSDT"
DAYS_TO_TEST = 7
TRAINING_WINDOW = 7
COLLATERAL = 10.0
LEVERAGE = 2.0
CAPITAL = 100.0
FIXED_STOP_LOSS = 0.04 # 4% (The proven winner from previous test)

def fetch_history(days):
    total_days = days + TRAINING_WINDOW + 2
    limit = 1000
    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval=15m&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'vol', 'x','y','z','a','b','c'])
    df['close'] = df['close'].astype(float)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df

def fetch_fng_history():
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

def calculate_slope(closes):
    if len(closes) < 21: return 0
    trend = savgol_filter(closes, 21, 3)
    return trend[-1] - trend[-3]

def run_optimizer():
    # 1. PREPARE DATA ONCE
    print(f"📥 Downloading Data for Optimization...")
    df = fetch_history(DAYS_TO_TEST)
    fng_map = fetch_fng_history()
    
    results = []
    
    print(f"🔄 Running 50 Simulations (TP 1% - 50%)...")
    
    # 2. THE LOOP (1% to 50%)
    for tp_percent in range(1, 51):
        tp_decimal = tp_percent / 100.0
        
        balance = CAPITAL
        trades = 0
        wins = 0
        active_trade = None
        
        start_index = len(df) - (DAYS_TO_TEST * 24 * 4) 
        
        # Fast Loop through candles
        for i in range(start_index, len(df)):
            current_price = df.iloc[i]['close']
            date_str = df.iloc[i]['time'].strftime('%Y-%m-%d')
            
            if active_trade is None:
                # CHECK ENTRY
                fng = fng_map.get(date_str, 50)
                past_slice = df.iloc[i-100:i+1]['close'].values
                slope = calculate_slope(past_slice)
                
                if fng < 40 and slope > 0:
                    active_trade = {
                        'entry': current_price,
                        'stop': current_price * (1 - FIXED_STOP_LOSS),
                        'target': current_price * (1 + tp_decimal)
                    }
            else:
                # CHECK EXIT
                if current_price >= active_trade['target']:
                    profit = (COLLATERAL * LEVERAGE) * tp_decimal
                    balance += profit
                    active_trade = None
                    trades += 1
                    wins += 1
                elif current_price <= active_trade['stop']:
                    loss = (COLLATERAL * LEVERAGE) * FIXED_STOP_LOSS
                    balance -= loss
                    active_trade = None
                    trades += 1
        
        # Store Result
        net_profit = balance - CAPITAL
        results.append({
            "tp": tp_percent,
            "profit": net_profit,
            "trades": trades,
            "wins": wins
        })

    # 3. RANKING THE WINNERS
    # Sort by Net Profit (Descending)
    sorted_results = sorted(results, key=lambda x: x['profit'], reverse=True)
    
    print("\n🏆 TOP 5 PROFITABLE SETTINGS (Past 7 Days)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"{'TP %':<10} {'PROFIT':<10} {'TRADES':<10} {'WIN RATE'}")
    print("----------------------------------------")
    
    for r in sorted_results[:5]:
        win_rate = (r['wins'] / r['trades'] * 100) if r['trades'] > 0 else 0
        print(f"{r['tp']}%{'':<8} ${r['profit']:<9.2f} {r['trades']:<10} {win_rate:.0f}%")
        
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Also find the 'Safest' (Most Consistent)
    # Sort by Win Rate (min 3 trades)
    safe_results = [r for r in results if r['trades'] >= 3]
    safe_results = sorted(safe_results, key=lambda x: (x['wins']/x['trades']), reverse=True)
    
    if safe_results:
        best_safe = safe_results[0]
        safe_wr = (best_safe['wins']/best_safe['trades'])*100
        print(f"\n🛡️ SAFEST SETTING: {best_safe['tp']}% TP (Win Rate: {safe_wr:.0f}%)")

if __name__ == "__main__":
    run_optimizer()
