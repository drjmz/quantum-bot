import requests
import numpy as np

def get_psychological_levels(current_price, step=100):
    """Finds the nearest 'Round Numbers' (e.g., 2900, 3000)"""
    level_below = (current_price // step) * step
    level_above = level_below + step
    return level_below, level_above

def fetch_liquidity_walls(symbol="ETHUSDT", limit=500):
    """
    Fetches the Order Book to find real 'Walls' (High Volume Nodes).
    Returns the price of the heaviest resistance (Ask) and support (Bid).
    """
    try:
        url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
        data = requests.get(url, timeout=5).json()
        
        bids = np.array(data['bids'], dtype=float) # Buying Support
        asks = np.array(data['asks'], dtype=float) # Selling Resistance
        
        # Logic: Find the price level with the massive volume spike
        # We group them into 'buckets' (e.g., $10 chunks) to see density
        
        # 1. Find Resistance (Ask Wall)
        # Simple approach: Find the single price with the Max Volume
        max_ask_idx = np.argmax(asks[:, 1])
        resistance_wall = asks[max_ask_idx, 0]
        
        # 2. Find Support (Bid Wall)
        max_bid_idx = np.argmax(bids[:, 1])
        support_wall = bids[max_bid_idx, 0]
        
        return support_wall, resistance_wall
    except Exception as e:
        print(f"⚠️ Orderbook Scan Failed: {e}")
        return 0, 999999

def adjust_tp_sl(action_type, current_price, raw_tp, raw_sl):
    """
    Smart Adjustment:
    If your Target is ABOVE a Wall, lower it to be BELOW the wall.
    """
    support_wall, resistance_wall = fetch_liquidity_walls()
    psy_support, psy_resistance = get_psychological_levels(current_price)
    
    final_tp = raw_tp
    
    if "LONG" in action_type:
        # LONG TP Logic:
        # 1. Check Real Order Book Wall
        if raw_tp > resistance_wall > current_price:
            print(f"🧱 WALL DETECTED at ${resistance_wall:.2f}. Lowering TP.")
            final_tp = min(final_tp, resistance_wall * 0.999) # Front-run by 0.1%
            
        # 2. Check Psychological Wall ($3000)
        if raw_tp > psy_resistance > current_price:
            print(f"🧠 PSYCH LEVEL at ${psy_resistance:.2f}. Lowering TP.")
            final_tp = min(final_tp, psy_resistance * 0.998) # Front-run by 0.2%

    elif "SHORT" in action_type:
        # SHORT TP Logic:
        # 1. Check Real Support Wall
        if raw_tp < support_wall < current_price:
            print(f"🧱 FLOOR DETECTED at ${support_wall:.2f}. Raising TP.")
            final_tp = max(final_tp, support_wall * 1.001)
            
        # 2. Check Psychological Floor
        if raw_tp < psy_support < current_price:
            print(f"🧠 PSYCH FLOOR at ${psy_support:.2f}. Raising TP.")
            final_tp = max(final_tp, psy_support * 1.002)
            
    return final_tp, raw_sl
