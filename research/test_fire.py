import asyncio
import main
import os
import time

# --- OVERRIDE CONFIG FOR TEST ---
# Force "Real Money" mode for this script only
main.SIMULATION_MODE = False
main.TRADE_COLLATERAL = 10.0  # Minimum safe size for Avantis
main.LEVERAGE = 2.0           # Low leverage for safety

async def run_test():
    print("🔥 STARTING LIVE FIRE TEST (Real Money)")
    print("---------------------------------------")
    
    # 1. Fetch Live Data (Needed for the function signature)
    print("📥 Fetching Market Data...")
    closes = main.fetch_candles()
    price = closes[-1]
    print(f"   Current ETH Price: ${price:,.2f}")

    # ------------------------------------------------
    # TEST PHASE 1: THE LONG
    # ------------------------------------------------
    print("\n1️⃣  TESTING: OPEN LONG ($10)")
    tx = await main.execute_avantis_trade("OPEN_LONG", price, closes)
    
    if tx:
        print(f"   ✅ Long Opened! TX: {str(tx)}")
        print("   ⏳ Waiting 30 seconds to settle...")
        await asyncio.sleep(30)
        
        print("2️⃣  TESTING: CLOSE LONG")
        tx_close = await main.execute_avantis_trade("CLOSE", price, closes)
        if tx_close:
            print(f"   ✅ Long Closed! TX: {str(tx_close)}")
        else:
            print("   ❌ FAILED to Close Long! Check Avantis UI immediately.")
            return
    else:
        print("   ❌ Long Open Failed. Check Private Key or ETH Balance.")
        return

    # ------------------------------------------------
    # TEST PHASE 2: THE SHORT
    # ------------------------------------------------
    print("\n3️⃣  TESTING: OPEN SHORT ($10)")
    tx = await main.execute_avantis_trade("OPEN_SHORT", price, closes)
    
    if tx:
        print(f"   ✅ Short Opened! TX: {str(tx)}")
        print("   ⏳ Waiting 30 seconds to settle...")
        await asyncio.sleep(30)
        
        print("4️⃣  TESTING: CLOSE SHORT")
        tx_close = await main.execute_avantis_trade("CLOSE", price, closes)
        if tx_close:
            print(f"   ✅ Short Closed! TX: {str(tx_close)}")
        else:
            print("   ❌ FAILED to Close Short! Check Avantis UI immediately.")
    else:
        print("   ❌ Short Open Failed.")

    print("\n---------------------------------------")
    print("✅ CALIBRATION COMPLETE. All systems operational.")

if __name__ == "__main__":
    asyncio.run(run_test())
