import requests
import pandas as pd
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
# Lighter.xyz API Endpoint (Arbitrum One)
BASE_URL = "https://api.lighter.xyz"
SYMBOL_ID = 1  # Usually 0 or 1 for ETH-USDC. We will fetch the map first to be sure.

def get_market_map():
    """Finds the correct ID for ETH-USDC"""
    try:
        url = f"{BASE_URL}/api/v1/orderBooks"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()
        
        # Look for ETH-USDC pair
        for market in data:
            if "ETH" in market['symbol'] and "USDC" in market['symbol']:
                print(f"✅ Found Market: {market['symbol']} (ID: {market['id']})")
                return market['id']
        
        print("⚠️ Could not find ETH-USDC on Lighter. Defaulting to ID 1.")
        return 1
    except Exception as e:
        print(f"❌ API Error (Markets): {e}")
        return 1

def fetch_lighter_walls(market_id):
    """Fetches the Order Book and detects walls"""
    url = f"{BASE_URL}/api/v1/orderBookDetails?id={market_id}"
    
    try:
        print(f"\n📡 Fetching Lighter.xyz Order Book (ID: {market_id})...")
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()

        # Parse Bids and Asks
        # Lighter returns data as strings usually, need to convert
        bids = pd.DataFrame(data['bids'], columns=['price', 'size'])
        asks = pd.DataFrame(data['asks'], columns=['price', 'size'])

        bids = bids.astype(float)
        asks = asks.astype(float)

        # 1. FIND THE WALLS (Highest Liquidity Levels)
        # We look for the price level with the single largest size
        max_bid_idx = bids['size'].idxmax()
        max_ask_idx = asks['size'].idxmax()

        support_wall_price = bids.iloc[max_bid_idx]['price']
        support_wall_size = bids.iloc[max_bid_idx]['size']

        resist_wall_price = asks.iloc[max_ask_idx]['price']
        resist_wall_size = asks.iloc[max_ask_idx]['size']

        # 2. CALCULATE "WALL STRENGTH" (Whale vs Retail)
        # If the wall is > 5x the average order size, it's a Whale Wall.
        avg_bid_size = bids['size'].mean()
        bid_strength = support_wall_size / avg_bid_size

        print("\n📊 LIGHTER.XYZ (ARBITRUM) LIQUIDITY REPORT")
        print("==========================================")
        print(f"📉 BEST BID:   ${bids.iloc[0]['price']:,.2f}")
        print(f"📈 BEST ASK:   ${asks.iloc[0]['price']:,.2f}")
        print("------------------------------------------")
        print(f"🧱 BUY WALL:   ${support_wall_price:,.2f}  (Size: {support_wall_size:.2f} ETH)")
        print(f"   ↳ Strength: {bid_strength:.1f}x average")
        print(f"🧱 SELL WALL:  ${resist_wall_price:,.2f} (Size: {resist_wall_size:.2f} ETH)")
        print("==========================================")

        return support_wall_price, resist_wall_price

    except Exception as e:
        print(f"❌ API Error (Orderbook): {e}")
        return 0, 0

if __name__ == "__main__":
    market_id = get_market_map()
    fetch_lighter_walls(market_id)
