import asyncio
import json
import os
import requests
import numpy as np
import joblib
import csv
from scipy.signal import savgol_filter
from datetime import datetime, timedelta
try: from telegram import Bot
except ImportError: Bot = None

from avantis_trader_sdk import TraderClient
from avantis_trader_sdk.types import TradeInput, TradeInputOrderType
from learner import run_learning_cycle
from analyst import generate_trade_analysis

# --- CONFIGURATION ---
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "True") == "True"
PRIVATE_KEY = os.getenv("AVANTIS_PRIVATE_KEY")
STATE_FILE = os.getenv("STATE_FILE", "data/bot_state.json")
SIGNAL_LOG_FILE = os.getenv("SIGNAL_LOG_FILE", "data/signals.csv")
MODEL_FILE = "data/sl_tp_model.pkl"
SYMBOL = "ETHUSDT"
TIMEFRAME = "4h"
BASE_RPC = "https://mainnet.base.org"
TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- SETTINGS ---
TRADE_COLLATERAL = 10.0
LEVERAGE = 5.0
PAIR_INDEX = 0
DEFAULT_SL = 0.02
DEFAULT_TP = 0.03
CONFIRMATION_MINUTES = 5

# --- DATA FETCHING ---
def fetch_candles():
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval={TIMEFRAME}&limit=100"
        data = requests.get(url).json()
        return np.array([float(x[4]) for x in data]), np.array([float(x[3]) for x in data])
    except: return np.array([]), np.array([])

def fetch_smart_money_data():
    ls_ratio = 1.0
    oi = 0.0
    try:
        url_ls = f"https://fapi.binance.com/futures/data/topLongShortAccountRatio?symbol={SYMBOL}&period=5m&limit=1"
        data_ls = requests.get(url_ls).json()
        if data_ls: ls_ratio = float(data_ls[0]['longShortRatio'])
        url_oi = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={SYMBOL}"
        data_oi = requests.get(url_oi).json()
        if data_oi: oi = float(data_oi['openInterest'])
    except: pass
    return ls_ratio, oi

def get_sentiment():
    try:
        fng = requests.get("https://api.alternative.me/fng/?limit=1").json()
        return int(fng['data'][0]['value'])
    except: return 50

# --- LOGIC ---
def fetch_liquidity_walls(limit=500):
    try:
        url = f"https://api.binance.com/api/v3/depth?symbol={SYMBOL}&limit={limit}"
        data = requests.get(url, timeout=5).json()
        bids = np.array(data['bids'], dtype=float)
        asks = np.array(data['asks'], dtype=float)
        return bids[np.argmax(bids[:, 1]), 0], asks[np.argmax(asks[:, 1]), 0]
    except: return 0, 999999

def calculate_fib_levels(closes):
    if len(closes) < 50: return {}
    high = np.max(closes)
    low = np.min(closes)
    diff = high - low
    return {0.5: high-(0.5*diff), 0.618: high-(0.618*diff), 0.786: high-(0.786*diff)}

def adjust_smart_targets(signal_type, current_price, raw_tp, raw_sl):
    support_wall, resistance_wall = fetch_liquidity_walls()
    
    final_tp = raw_tp
    final_sl = raw_sl
    
    if "LONG" in signal_type:
        # 1. TP LOGIC: Front-run the Sell Wall (Sell before the crowd)
        if raw_tp > resistance_wall > current_price: 
            final_tp = min(final_tp, resistance_wall * 0.999)
            
        # 2. SL LOGIC: Hide behind the Buy Wall (Use wall as a shield)
        # If the wall is between our Entry and our calculated SL, tighten the SL 
        # to just below the wall.
        if current_price > support_wall > raw_sl:
            final_sl = max(final_sl, support_wall * 0.995) # 0.5% buffer below wall

    elif "SHORT" in signal_type:
        # 1. TP LOGIC: Front-run the Buy Wall (Buy back before the crowd)
        if raw_tp < support_wall < current_price: 
            final_tp = max(final_tp, support_wall * 1.001)
            
        # 2. SL LOGIC: Hide behind the Sell Wall
        if current_price < resistance_wall < raw_sl:
             final_sl = min(final_sl, resistance_wall * 1.005) # 0.5% buffer above wall

    return final_tp, final_sl

def calculate_quantum_wave(closes):
    if len(closes) < 21: return 0, 0
    trend = savgol_filter(closes, 21, 3)
    slope = trend[-1] - trend[-2]
    spread = abs(closes[-1] - trend[-1]) / trend[-1]
    return slope, spread

def analyze_flow_state(slope, current_oi, prev_oi):
    if prev_oi == 0: return "Neutral", 0
    oi_change = ((current_oi - prev_oi) / prev_oi) * 100
    if slope > 0 and oi_change > 0: return "REAL BREAKOUT", oi_change
    if slope > 0 and oi_change < 0: return "SHORT SQUEEZE (Fakeout)", oi_change
    if slope < 0 and oi_change > 0: return "REAL DUMP", oi_change
    if slope < 0 and oi_change < 0: return "LONG LIQUIDATION", oi_change
    return "Choppy", oi_change

def calc_features_for_ai(prices, lows):
    if len(prices) < 50: return None
    smooth = savgol_filter(prices, 21, 3)
    slope = smooth[-1] - smooth[-2]
    volatility = np.std(prices[-20:])
    delta = np.diff(prices)
    gain = delta[delta > 0].mean() if len(delta[delta > 0]) > 0 else 0
    loss = -delta[delta < 0].mean() if len(delta[delta < 0]) > 0 else 1
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    recent_low = np.min(lows[-50:])
    dist = (prices[-1] - recent_low) / prices[-1]
    return [slope, volatility, rsi, dist]

def get_ai_parameters(closes, lows):
    if not os.path.exists(MODEL_FILE): return DEFAULT_SL, DEFAULT_TP
    try:
        model = joblib.load(MODEL_FILE)
        features = calc_features_for_ai(closes, lows)
        if not features: return DEFAULT_SL, DEFAULT_TP
        prediction = model.predict([features])[0]
        return max(0.005, min(prediction[0], 0.10)), max(0.01, min(prediction[1], 0.20))
    except: return DEFAULT_SL, DEFAULT_TP

# --- STATE MANAGEMENT (NOW WITH PERSISTENCE) ---
def update_state(status, is_open, entry, price, fng, slope, spread, win_prob, whale_ratio, decision, reason, suggest_sl, suggest_tp, flow_state, oi_delta, pending_info, active_signal_type, signal_start_time):
    # Convert signal_start_time to string if it exists
    start_str = signal_start_time.isoformat() if signal_start_time else None
    
    state = {
        "status": status, "is_open": is_open, "entry_price": entry,
        "current_price": price, "last_update": datetime.now().strftime("%H:%M:%S"),
        "sentiment": fng, "slope": slope, "spread": spread,
        "win_probability": win_prob, "whale_ratio": whale_ratio,
        "decision": decision, "reason": reason,
        "suggested_sl": suggest_sl, "suggested_tp": suggest_tp,
        "leverage": LEVERAGE, "flow_state": flow_state, "oi_delta": oi_delta,
        "pending_signal": pending_info,
        # SAVING CRITICAL SIGNAL DATA
        "active_signal_type": active_signal_type,
        "signal_start_time": start_str
    }
    with open(STATE_FILE, "w") as f: json.dump(state, f)

def load_state():
    if os.path.exists(STATE_FILE):
        try: return json.load(open(STATE_FILE))
        except: pass
    return None

def log_signal_to_history(signal_type, price, sl, tp, conf, reason):
    file_exists = os.path.isfile(SIGNAL_LOG_FILE)
    if file_exists:
        with open(SIGNAL_LOG_FILE, "r") as f:
            lines = f.readlines()
            if lines and len(lines[-1].split(',')) > 1:
                 if lines[-1].split(',')[1] == signal_type and (datetime.now() - datetime.strptime(lines[-1].split(',')[0], "%Y-%m-%d %H:%M:%S")).seconds < 3600: return
    with open(SIGNAL_LOG_FILE, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists: writer.writerow(["Timestamp", "Type", "Price", "Stop Loss", "Take Profit", "AI Conf", "Reason"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), signal_type, f"${price:.2f}", f"${sl:.2f}", f"${tp:.2f}", f"{conf:.1f}%", reason])

async def send_telegram_alert(msg_type, price, tp, sl, reason, slope=0, whale=0, fng=50, flow="N/A", win_prob=0):
    if not TG_TOKEN or not TG_CHAT_ID or Bot is None: return
    try:
        print("🤔 Consulting AI Analyst...")
        ai_summary = generate_trade_analysis(msg_type, price, slope, whale, fng, flow, win_prob)
        bot = Bot(token=TG_TOKEN)

        risk = abs(price - sl) / price * 100 if price else 0
        reward = abs(tp - price) / price * 100 if price else 0

        # STYLE 1: HEARTBEAT
        if "UPDATE" in msg_type or "HEARTBEAT" in msg_type:
            message = (
                f"📡 <b>SYSTEM HEARTBEAT | 4H SCAN</b>\n"
                f"<code>------------------------------</code>\n"
                f"💵 <b>Price:</b> ${price:,.2f}\n"
                f"📉 <b>Slope:</b> {slope:.2f} | 🐳 <b>Whale:</b> {whale:.2f}\n"
                f"🧠 <b>AI Confidence:</b> {win_prob:.1f}%\n\n"
                f"🔭 <i>Hypothetical Targets:</i>\n"
                f"Target: ${tp:,.2f} | Stop: ${sl:,.2f}\n\n"
                f"🤖 <b>AI Log:</b> {ai_summary}"
            )

        # STYLE 2: TRADE SIGNAL
        else:
            emoji = "🚫" if "INVALID" in msg_type else "🚨" if "SHORT" in msg_type else "🚀"
            header = f"⚠️ <b>ACTION REQUIRED: {msg_type}</b> ⚠️"
            
            message = (
                f"{emoji} {header}\n\n"
                f"💵 <b>ENTRY: ${price:,.2f}</b>\n"
                f"<code>------------------------------</code>\n"
                f"🎯 <b>TARGET:</b> ${tp:,.2f} (+{reward:.1f}%)\n"
                f"🛑 <b>STOP:</b>   ${sl:,.2f} (-{risk:.1f}%)\n"
                f"<code>------------------------------</code>\n\n"
                f"🔥 <b>THE LOGIC:</b>\n"
                f"{reason}\n\n"
                f"🤖 <b>AI ANALYST INSIGHT:</b>\n"
                f"<i>{ai_summary}</i>\n\n"
                f"📊 <b>Market Techs:</b>\n"
                f"Flow: {flow} | Sentiment: {fng}"
            )
        await bot.send_message(chat_id=TG_CHAT_ID, text=message, parse_mode='HTML')
    except Exception as e: print(f"⚠️ Telegram Error: {e}")

async def execute_avantis_trade(action_type, current_price, sl_price, tp_price):
    if SIMULATION_MODE: return "0xSimulatedHash"
    if not PRIVATE_KEY: return None
    try:
        client = TraderClient(BASE_RPC)
        client.set_local_signer(PRIVATE_KEY)
        is_long = "LONG" in action_type
        if "OPEN" in action_type:
            trade_input = TradeInput(pair_index=PAIR_INDEX, is_long=is_long, collateral=TRADE_COLLATERAL, leverage=LEVERAGE, slippage=0.02, tp=tp_price, sl=sl_price)
            tx = await client.trade.open_market_trade(trade_input, TradeInputOrderType.MARKET)
            return tx.transaction_hash
        elif action_type == "CLOSE":
            positions = await client.trade.get_positions(client.signer.address)
            if positions: return (await client.trade.close_trade(positions[0].id)).transaction_hash
    except Exception as e: print(f"❌ Execution Error: {e}"); return None

async def main():
    print(f"🧠 Quantum v9.7 (Safe Persistence Patch) | Mode: {'SIMULATION' if SIMULATION_MODE else 'REAL'}")

    old_state = load_state()
    is_in_position = old_state.get("is_open", False) if old_state else False
    entry_price = old_state.get("entry_price", 0) if old_state else 0

    # RESTORE STATE IF EXISTS
    active_signal_type = old_state.get("active_signal_type", None) if old_state else None
    signal_start_str = old_state.get("signal_start_time", None) if old_state else None
    signal_start_time = datetime.fromisoformat(signal_start_str) if signal_start_str else None
    
    # Restoring specific targets if they exist in JSON, otherwise 0
    signal_tp = old_state.get("suggested_tp", 0) if old_state else 0
    signal_sl = old_state.get("suggested_sl", 0) if old_state else 0

    current_win_prob = run_learning_cycle()
    last_learning = datetime.now()
    last_alert = datetime.min
    prev_oi = 0
    boot_time = datetime.now()

    status_freeze_until = None
    frozen_decision = ""
    frozen_reason = ""
    
    was_in_position = is_in_position

    while True:
        try:
            if (datetime.now() - boot_time).seconds < 60: 
                fetch_smart_money_data(); await asyncio.sleep(5); continue

            # --- 1. ACCELERATED LEARNING ---
            if (datetime.now() - last_learning).seconds > 3600:
                print("🧠 RE-TRAINING AI MODEL (Hourly Cycle)...")
                current_win_prob = run_learning_cycle()
                last_learning = datetime.now()

            closes, lows = fetch_candles()
            price = closes[-1] if len(closes) > 0 else 0
            fng = get_sentiment()
            slope, vol = calculate_quantum_wave(closes)
            whale_ratio, current_oi = fetch_smart_money_data()
            flow_state, oi_delta = analyze_flow_state(slope, current_oi, prev_oi)
            prev_oi = current_oi

            ai_sl_pct, ai_tp_pct = get_ai_parameters(closes, lows)
            raw_tp = price * (1 + ai_tp_pct)
            raw_sl = price * (1 - ai_sl_pct)

            # --- 2. HOURLY ALERT ---
            if (datetime.now() - last_alert).seconds > 3600:
                await send_telegram_alert(
                    "SYSTEM HEARTBEAT", price, raw_tp, raw_sl, "Hourly Heartbeat", 
                    slope, whale_ratio, fng, flow_state, current_win_prob
                )
                last_alert = datetime.now()

            decision = "SCANNING"
            reason = f"Slope: {slope:.4f} | Whales: {whale_ratio:.2f}"
            pending_info = None

            # --- 3. POSITION MANAGEMENT ---
            if is_in_position:
                was_in_position = True
                pnl = abs((price - entry_price) / entry_price) * 100
                decision = "HOLDING"
                reason = f"On-Chain Protection Active. PnL: {pnl:.2f}%"
            
            elif was_in_position and not is_in_position:
                print("🎓 Trade Closed. Initiating IMMEDIATE Learning Cycle...")
                await asyncio.sleep(10)
                current_win_prob = run_learning_cycle()
                last_learning = datetime.now()
                was_in_position = False

            # --- 4. STATUS FREEZE & INVALIDATION ---
            if status_freeze_until and datetime.now() < status_freeze_until:
                decision = frozen_decision
                reason = frozen_reason

            elif active_signal_type and not is_in_position:
                if signal_start_time:
                    elapsed_mins = (datetime.now() - signal_start_time).seconds / 60
                else:
                    elapsed_mins = 0

                is_invalid = False
                invalid_reason = ""

                if elapsed_mins > 30: is_invalid = True; invalid_reason = "Timed Out (30m)"
                if active_signal_type == "LONG" and slope < -0.5: is_invalid = True; invalid_reason = "Trend Broken"
                if active_signal_type == "SHORT" and slope > 0.5: is_invalid = True; invalid_reason = "Trend Broken"

                if is_invalid:
                    print(f"🚫 {invalid_reason}")
                    log_signal_to_history(f"CANCEL {active_signal_type}", price, 0, 0, current_win_prob, invalid_reason)
                    await send_telegram_alert(f"{active_signal_type} INVALIDATED", price, 0, 0, invalid_reason, slope, whale_ratio, fng, flow_state, current_win_prob)

                    status_freeze_until = datetime.now() + timedelta(minutes=15)
                    frozen_decision = "🚫 SIGNAL INVALIDATED"
                    frozen_reason = f"Reason: {invalid_reason}"
                    active_signal_type = None; signal_start_time = None
                else:
                    decision = f"🚀 {active_signal_type} ACTIVE ({int(elapsed_mins)}m)"
                    reason = "Signal Valid. Waiting for Manual Entry."

            # --- 5. SIGNAL SCAN ---
            elif not is_in_position and not status_freeze_until:
                trend_up = slope > 0.5
                whale_buy = (whale_ratio > 1.2) and (fng < 40)
                fibs = calculate_fib_levels(closes)
                near_support = (abs(price - fibs[0.618])/price < 0.005) or (abs(price - fibs[0.786])/price < 0.005)

                detected = None
                if trend_up and (fng < 60 or whale_buy): detected = "LONG"
                elif near_support and slope > 0.1 and fng < 30: detected = "LONG"

                if detected:
                    req_conf = 30 if whale_buy else 40
                    if current_win_prob > req_conf:
                        smart_tp, smart_sl, adj_note = adjust_smart_targets(detected, price, raw_tp, raw_sl)

                        if active_signal_type != detected:
                            active_signal_type = detected
                            signal_start_time = datetime.now()
                            signal_tp = smart_tp
                            signal_sl = smart_sl

                        decision = f"🚀 {detected} SIGNAL"
                        reason = f"Trend/Bounce + AI ({current_win_prob:.0f}%)"

                        log_signal_to_history(detected, price, smart_sl, smart_tp, current_win_prob, reason)
                        
                        await send_telegram_alert(
                            f"{detected} ETH", price, smart_tp, smart_sl, reason, 
                            slope, whale_ratio, fng, flow_state, current_win_prob
                        )

            # --- 6. DISPLAY SYNC (FIXED FOR 0.00 ISSUE) ---
            # If we are in a trade, we KEEP the original signal TP/SL.
            # If we are scanning, we show the raw market TP/SL.
            # If active_signal_type was restored, use its values.
            # Fallback: If in position but TP is 0 (lost data), show raw targets.
            
            if is_in_position and signal_tp == 0:
                 # Lost persistence case: Show current dynamic targets
                 display_sl = raw_sl
                 display_tp = raw_tp
            elif active_signal_type or is_in_position:
                 display_sl = signal_sl
                 display_tp = signal_tp
            else:
                 display_sl = raw_sl
                 display_tp = raw_tp

            update_state("ACTIVE", is_in_position, entry_price, price, fng, slope, vol, current_win_prob, whale_ratio, decision, reason, display_sl, display_tp, flow_state, oi_delta, pending_info, active_signal_type, signal_start_time)
            await asyncio.sleep(60)

        except Exception as e: print(f"Loop Error: {e}"); await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
