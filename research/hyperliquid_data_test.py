import requests
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
API_URL = "https://api.hyperliquid.xyz/info"
COIN = "ETH"

def fetch_hyperliquid_book():
    print(f"\n📡 Connecting to Hyperliquid (Mainnet)...")
    
    payload = {"type": "l2Book", "coin": COIN}
    
    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        levels = data['levels']
        
        # 1. LOAD DATA (Native Keys: px=price, sz=size, n=orders)
        bids = pd.DataFrame(levels[0])
        asks = pd.DataFrame(levels[1])

        # 2. RENAME COLUMNS (To match our logic)
        bids = bids.rename(columns={"px": "price", "sz": "size", "n": "orders"})
        asks = asks.rename(columns={"px": "price", "sz": "size", "n": "orders"})
        
        # 3. CONVERT TYPES (Strings -> Floats)
        bids = bids.astype(float)
        asks = asks.astype(float)
        
        return bids, asks

    except Exception as e:
        print(f"❌ API Error: {e}")
        return pd.DataFrame(), pd.DataFrame()

def analyze_walls(bids, asks):
    if bids.empty or asks.empty:
        print("⚠️ No data received.")
        return

    # Find the single biggest order levels
    max_bid_idx = bids['size'].idxmax()
    max_ask_idx = asks['size'].idxmax()

    support_wall_price = bids.iloc[max_bid_idx]['price']
    support_wall_size = bids.iloc[max_bid_idx]['size']

    resist_wall_price = asks.iloc[max_ask_idx]['price']
    resist_wall_size = asks.iloc[max_ask_idx]['size']

    # Calculate Strength (Wall Size vs Average)
    avg_bid_size = bids['size'].mean()
    bid_strength = support_wall_size / avg_bid_size if avg_bid_size > 0 else 0

    avg_ask_size = asks['size'].mean()
    ask_strength = resist_wall_size / avg_ask_size if avg_ask_size > 0 else 0

    current_price = (bids.iloc[0]['price'] + asks.iloc[0]['price']) / 2

    print(f"\n📊 HYPERLIQUID WALL REPORT ({COIN})")
    print("==========================================")
    print(f"💵 Current Price: ${current_price:,.2f}")
    print("------------------------------------------")
    print(f"🧱 BUY WALL:   ${support_wall_price:,.2f}")
    print(f"   ↳ Size:     {support_wall_size:.2f} ETH")
    print(f"   ↳ Strength: {bid_strength:.1f}x")
    print("------------------------------------------")
    print(f"🧱 SELL WALL:  ${resist_wall_price:,.2f}")
    print(f"   ↳ Size:     {resist_wall_size:.2f} ETH")
    print(f"   ↳ Strength: {ask_strength:.1f}x")
    print("==========================================")
    
    # Range Logic
    dist_down = current_price - support_wall_price
    dist_up = resist_wall_price - current_price
    
    print(f"\n📉 Distance to Support: ${dist_down:.2f}")
    print(f"📈 Distance to Resist:  ${dist_up:.2f}")

if __name__ == "__main__":
    bids, asks = fetch_hyperliquid_book()
    analyze_walls(bids, asks)
