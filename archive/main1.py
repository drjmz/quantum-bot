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

# --- CONFIGURATION ---
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "True") == "True"
PRIVATE_KEY = os.getenv("AVANTIS_PRIVATE_KEY")

# Use Degen-Specific Files (Set via Env in Docker, but defaults here just in case)
STATE_FILE = os.getenv("STATE_FILE", "data/degen_state.json")
SIGNAL_LOG_FILE = os.getenv("SIGNAL_LOG_FILE", "data/degen_signals.csv")
MODEL_FILE = "data/sl_tp_model.pkl" # Shared Brain is fine

SYMBOL = "ETHUSDT"
TIMEFRAME = "4h"
BASE_RPC = "https://mainnet.base.org" 
TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- DEGEN SETTINGS (50x Turbo) ---
TRADE_COLLATERAL = 10.0 
LEVERAGE = 50.0           # <--- 50x POWER
PAIR_INDEX = 0          
DEFAULT_SL = 0.009        # 0.9% Stop Loss (Tight for 50x)
DEFAULT_TP = 0.02         # 2.0% Take Profit (Quick Scalp)
CONFIRMATION_MINUTES = 5 
WARMUP_MINUTES = 5       

# --- DATA FETCHING ---
def fetch_candles():
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval={TIMEFRAME}&limit=100"
        data = requests.get(url).json()
        return np.array([float(x[4]) for x in data])
    except: return np.array([])

def fetch_smart_money_data():
    """Fetches BOTH Whale Ratio and Open Interest"""
    ls_ratio = 1.0
    oi = 0.0
    try:
        # 1. Top Trader Ratio
        url_ls = f"https://fapi.binance.com/futures/data/topLongShortAccountRatio?symbol={SYMBOL}&period=5m&limit=1"
        data_ls = requests.get(url_ls).json()
        if data_ls: ls_ratio = float(data_ls[0]['longShortRatio'])
        
        # 2. Open Interest
        url_oi = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={SYMBOL}"
        data_oi = requests.get(url_oi).json()
        if data_oi: oi = float(data_oi['openInterest'])
        
    except Exception as e: 
        print(f"⚠️ Data Fetch Error: {e}")
    
    return ls_ratio, oi

def get_sentiment():
    try:
        fng = requests.get("https://api.alternative.me/fng/?limit=1").json()
        return int(fng['data'][0]['value'])
    except: return 50

# --- ORDER BOOK ENGINE ---
def get_psychological_levels(current_price, step=100):
    level_below = (current_price // step) * step
    level_above = level_below + step
    return level_below, level_above

def fetch_liquidity_walls(limit=500):
    try:
        url = f"https://api.binance.com/api/v3/depth?symbol={SYMBOL}&limit={limit}"
        data = requests.get(url, timeout=5).json()
        bids = np.array(data['bids'], dtype=float)
        asks = np.array(data['asks'], dtype=float)
        
        max_ask_idx = np.argmax(asks[:, 1])
        resistance_wall = asks[max_ask_idx, 0]
        
        max_bid_idx = np.argmax(bids[:, 1])
        support_wall = bids[max_bid_idx, 0]
        return support_wall, resistance_wall
    except: return 0, 999999

def adjust_smart_targets(signal_type, current_price, raw_tp, raw_sl):
    support_wall, resistance_wall = fetch_liquidity_walls()
    psy_support, psy_resistance = get_psychological_levels(current_price)
    
    final_tp = raw_tp
    final_sl = raw_sl
    adjust_reason = ""

    if "LONG" in signal_type:
        if raw_tp > resistance_wall > current_price:
            final_tp = min(final_tp, resistance_wall * 0.999)
            adjust_reason += f"TP < Wall (${resistance_wall:.0f}). "
        if raw_tp > psy_resistance > current_price:
            final_tp = min(final_tp, psy_resistance * 0.998)
            adjust_reason += f"TP < Psych (${psy_resistance:.0f}). "
            
    elif "SHORT" in signal_type:
        if raw_tp < support_wall < current_price:
            final_tp = max(final_tp, support_wall * 1.001)
            adjust_reason += f"TP > Wall (${support_wall:.0f}). "
        if raw_tp < psy_support < current_price:
            final_tp = max(final_tp, psy_support * 1.002)
            adjust_reason += f"TP > Psych (${psy_support:.0f}). "

    return final_tp, final_sl, adjust_reason

# --- LOGIC ENGINE ---
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

def calc_features_for_ai(prices):
    if len(prices) < 50: return None
    smooth = savgol_filter(prices, 21, 3)
    slope = smooth[-1] - smooth[-2]
    volatility = np.std(prices[-20:])
    delta = np.diff(prices)
    gain = delta[delta > 0].mean() if len(delta[delta > 0]) > 0 else 0
    loss = -delta[delta < 0].mean() if len(delta[delta < 0]) > 0 else 1
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return [slope, volatility, rsi]

def get_ai_parameters(closes):
    # FOR DEGEN BOT: We prefer the Hardcoded Champion Settings (0.9% / 2%)
    # But we still check AI just to get a confidence score later
    return DEFAULT_SL, DEFAULT_TP

# --- STATE & EXECUTION ---
def update_state(status, is_open, entry, price, fng, slope, spread, win_prob, whale_ratio, decision, reason, suggest_sl, suggest_tp, flow_state, oi_delta, pending_signal=None):
    state = {
        "status": status, "is_open": is_open, "entry_price": entry,
        "current_price": price, "last_update": datetime.now().strftime("%H:%M:%S"),
        "sentiment": fng, "slope": slope, "spread": spread, 
        "win_probability": win_prob, "whale_ratio": whale_ratio,
        "decision": decision, "reason": reason,
        "suggested_sl": suggest_sl, "suggested_tp": suggest_tp,
        "leverage": LEVERAGE, "flow_state": flow_state, "oi_delta": oi_delta,
        "pending_signal": pending_signal 
    }
    with open(STATE_FILE, "w") as f: json.dump(state, f)

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

async def send_telegram_alert(msg_type, price, tp, sl, reason, adjust_note=""):
    if not TG_TOKEN or not TG_CHAT_ID or Bot is None: return
    try:
        bot = Bot(token=TG_TOKEN)
        emoji = "🚨" if "SHORT" in msg_type else "🚀"
        note = f"\n🧱 <b>Orderbook Adj:</b> {adjust_note}" if adjust_note else ""
        message = (f"{emoji} <b>DEGEN BOT SIGNAL: {msg_type}</b>\n\n💵 <b>Entry:</b> ${price:,.2f}\n🎯 <b>Target:</b> ${tp:,.2f}\n🛑 <b>Stop:</b> ${sl:,.2f}\n⚡ <b>Lev:</b> {LEVERAGE}x{note}\n\n🧠 <b>Logic:</b> {reason}")
        await bot.send_message(chat_id=TG_CHAT_ID, text=message, parse_mode='HTML')
    except Exception as e: print(f"⚠️ Telegram Error: {e}")

async def execute_avantis_trade(action_type, current_price, sl_price, tp_price):
    if SIMULATION_MODE: 
        print(f"🧪 SIMULATION: {action_type} @ ${current_price:.2f}")
        return "0xSimulatedHash"
    if not PRIVATE_KEY: return None
    try:
        client = TraderClient(BASE_RPC)
        client.set_local_signer(PRIVATE_KEY)
        
        is_long = "LONG" in action_type
        
        if "OPEN" in action_type:
            trade_input = TradeInput(
                pair_index=PAIR_INDEX,
                is_long=is_long,
                collateral=TRADE_COLLATERAL,
                leverage=LEVERAGE,
                slippage=0.02,
                tp=tp_price,
                sl=sl_price
            )
            tx = await client.trade.open_market_trade(trade_input, TradeInputOrderType.MARKET)
            return tx.transaction_hash
        elif action_type == "CLOSE":
            positions = await client.trade.get_positions(client.signer.address)
            if positions:
                tx = await client.trade.close_trade(positions[0].id)
                return tx.transaction_hash
    except Exception as e:
        print(f"❌ Execution Error: {e}")
        return None

async def main():
    print(f"🧠 Quantum v8.1 (Degen Edition) | Mode: {'SIMULATION' if SIMULATION_MODE else 'REAL MONEY'}")
    
    old_state = None
    if os.path.exists(STATE_FILE):
        try: old_state = json.load(open(STATE_FILE))
        except: pass
    is_in_position = old_state.get("is_open", False) if old_state else False
    entry_price = old_state.get("entry_price", 0) if old_state else 0
    
    current_win_prob = run_learning_cycle() 
    last_learning = datetime.now()
    last_alert = datetime.min
    prev_oi = 0
    signal_start = None
    signal_type = None
    
    boot_time = datetime.now()

    while True:
        try:
            if (datetime.now() - boot_time).seconds < 300:
                fetch_smart_money_data(); await asyncio.sleep(10); continue

            if (datetime.now() - last_learning).seconds > 14400:
                current_win_prob = run_learning_cycle(); last_learning = datetime.now()

            closes = fetch_candles()
            price = closes[-1] if len(closes) > 0 else 0
            fng = get_sentiment()
            slope, vol = calculate_quantum_wave(closes)
            whale_ratio, current_oi = fetch_smart_money_data()
            flow_state, oi_delta = analyze_flow_state(slope, current_oi, prev_oi)
            prev_oi = current_oi
            
            # Use Default High-Lev Settings (No AI override for Sizing)
            raw_tp = price * (1 + DEFAULT_TP)
            raw_sl = price * (1 - DEFAULT_SL)
            
            smart_tp, smart_sl, adj_note = adjust_smart_targets("LONG", price, raw_tp, raw_sl)
            
            decision = "SCANNING"
            reason = f"Slope: {slope:.4f} | Whales: {whale_ratio:.2f}"
            pending_info = None

            if not is_in_position:
                trend_up = slope > 0.5
                trend_down = slope < -0.5
                whale_buy = (whale_ratio > 1.2) and (fng < 40)
                
                safe_long = trend_up and (fng < 60 or whale_buy)
                safe_short = trend_down and (fng > 30)
                
                detected = "LONG" if safe_long else "SHORT" if safe_short else None
                
                if detected:
                    req_conf = 30 if whale_buy else 40
                    if current_win_prob > req_conf:
                        smart_tp, smart_sl, adj_note = adjust_smart_targets(detected, price, raw_tp, raw_sl)
                        
                        if signal_start is None or signal_type != detected:
                            signal_start = datetime.now()
                            signal_type = detected
                            print(f"⏳ {detected} detected. Confirming...")
                        
                        elapsed = (datetime.now() - signal_start).seconds / 60
                        pending_info = f"Validating {detected}... ({elapsed:.1f}/5m)"
                        
                        if elapsed >= CONFIRMATION_MINUTES:
                            decision = f"🚀 {detected} SIGNAL"
                            reason = f"Confirmed Trend + Whales ({whale_ratio:.2f}) + {adj_note}"
                            
                            log_signal_to_history(detected, price, smart_sl, smart_tp, current_win_prob, reason)
                            
                            if (datetime.now() - last_alert).seconds > 3600:
                                await send_telegram_alert(f"{detected} ETH", price, smart_tp, smart_sl, reason, adj_note)
                                last_alert = datetime.now()
                            
                            tx = await execute_avantis_trade(f"OPEN_{detected}", price, smart_sl, smart_tp)
                            if tx: is_in_position = True; entry_price = price; signal_start = None
                    else: signal_start = None
                else: signal_start = None

            else:
                pnl = abs((price - entry_price) / entry_price) * 100
                decision = "HOLDING"
                reason = f"On-Chain Protection Active. PnL: {pnl:.2f}%"

            update_state("ACTIVE", is_in_position, entry_price, price, fng, slope, vol, current_win_prob, whale_ratio, decision, reason, smart_sl, smart_tp, flow_state, oi_delta, pending_info)
            await asyncio.sleep(60)

        except Exception as e: print(f"Loop Error: {e}"); await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
