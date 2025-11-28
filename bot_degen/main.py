import asyncio
import json
import os
import requests
import numpy as np
import joblib
import csv
import pandas as pd
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
STATE_FILE = os.getenv("STATE_FILE", "data/degen_state.json")
SIGNAL_LOG_FILE = os.getenv("SIGNAL_LOG_FILE", "data/degen_signals.csv")
MODEL_FILE = "data/sl_tp_model.pkl"
SYMBOL = "ETHUSDT"
TIMEFRAME = "4h"
BASE_RPC = "https://mainnet.base.org"
TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TRADE_COLLATERAL = 10.0
LEVERAGE = 50.0
PAIR_INDEX = 0
DEFAULT_SL = 0.009
DEFAULT_TP = 0.02
CONFIRMATION_MINUTES = 5

def fetch_candles():
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval={TIMEFRAME}&limit=100"
        data = requests.get(url).json()
        return np.array([float(x[4]) for x in data])
    except: return np.array([])

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

# --- UPGRADED WALL DETECTION (RETURNS SIZES) ---
def fetch_liquidity_walls(limit=500):
    url = "https://api.hyperliquid.xyz/info"
    payload = {"type": "l2Book", "coin": "ETH"}
    try:
        response = requests.post(url, json=payload, timeout=2)
        data = response.json()
        if 'levels' not in data: return 0, 0, 999999, 0
        levels = data['levels']
        bids = pd.DataFrame(levels[0])
        asks = pd.DataFrame(levels[1])
        if bids.empty or asks.empty: return 0, 0, 999999, 0
        col_map = {"px": "price", "sz": "size", "n": "orders", 0: "price", 1: "size", 2: "orders"}
        bids = bids.rename(columns=col_map).astype(float)
        asks = asks.rename(columns=col_map).astype(float)
        if 'size' not in bids.columns: return 0, 0, 999999, 0
        max_bid_idx = bids['size'].idxmax()
        max_ask_idx = asks['size'].idxmax()
        buy_px = bids.iloc[max_bid_idx]['price']
        buy_sz = bids.iloc[max_bid_idx]['size']
        sell_px = asks.iloc[max_ask_idx]['price']
        sell_sz = asks.iloc[max_ask_idx]['size']
        current = (bids.iloc[0]['price'] + asks.iloc[0]['price']) / 2
        if abs(buy_px - current) / current > 0.05: buy_px = current * 0.99
        if abs(sell_px - current) / current > 0.05: sell_px = current * 1.01
        return buy_px, buy_sz, sell_px, sell_sz
    except Exception as e:
        print(f"⚠️ Hyperliquid API Error: {e}")
        return 0, 0, 999999, 0

def adjust_smart_targets(signal_type, current_price, raw_tp, raw_sl):
    buy_px, buy_sz, sell_px, sell_sz = fetch_liquidity_walls()
    final_tp = raw_tp
    final_sl = raw_sl
    
    if "LONG" in signal_type:
        if raw_tp > sell_px > current_price: 
            final_tp = min(final_tp, sell_px * 0.999)
        if current_price > buy_px > raw_sl:
            final_sl = max(final_sl, buy_px * 0.995)
    elif "SHORT" in signal_type:
        if raw_tp < buy_px < current_price: 
            final_tp = max(final_tp, buy_px * 1.001)
        if current_price < sell_px < raw_sl:
             final_sl = min(final_sl, sell_px * 1.005)
    return final_tp, final_sl, (buy_px, buy_sz, sell_px, sell_sz)

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

def get_ai_parameters(closes):
    return DEFAULT_SL, DEFAULT_TP

def update_state(status, is_open, entry, price, fng, slope, spread, win_prob, whale_ratio, decision, reason, suggest_sl, suggest_tp, flow_state, oi_delta, pending_info, active_signal_type, signal_start_time):
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
        "active_signal_type": active_signal_type, "signal_start_time": start_str 
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

# --- TELEGRAM ALERT WITH WALL INFO ---
async def send_telegram_alert(msg_type, price, tp, sl, reason, slope=0, whale=0, fng=50, flow="N/A", win_prob=0, walls=None):
    if not TG_TOKEN or not TG_CHAT_ID or Bot is None: return
    try:
        print("🤔 Consulting AI Analyst...")
        ai_summary = generate_trade_analysis(msg_type, price, slope, whale, fng, flow, win_prob)
        bot = Bot(token=TG_TOKEN)

        risk = abs(price - sl) / price * 100 if price else 0
        reward = abs(tp - price) / price * 100 if price else 0

        wall_msg = ""
        if walls and len(walls) == 4:
            b_px, b_sz, s_px, s_sz = walls
            wall_msg = (
                f"🧱 <b>Liquidity Walls:</b>\n"
                f"🟢 Buy: ${b_px:,.2f} ({b_sz:.0f} ETH)\n"
                f"🔴 Sell: ${s_px:,.2f} ({s_sz:.0f} ETH)\n"
            )

        if "UPDATE" in msg_type or "HEARTBEAT" in msg_type:
            message = (
                f"📡 <b>DEGEN RADAR | SCANNING...</b>\n"
                f"<code>-------------------------</code>\n"
                f"💵 <b>Price:</b> ${price:,.2f}\n"
                f"📊 <b>Techs:</b> Slope {slope:.2f} | Whales {whale:.2f}\n"
                f"🎲 <b>Win Prob:</b> {win_prob:.1f}%\n\n"
                f"💭 <i>If we ape now (Hypothetical):</i>\n"
                f"TP: ${tp:,.2f} | SL: ${sl:,.2f}\n"
                f"<code>-------------------------</code>\n\n"
                f"{wall_msg}\n"
                f"🤖 <b>AI Roast:</b>\n"
                f"<i>{ai_summary}</i>"
            )
        else:
            emoji = "🚫" if "INVALID" in msg_type else "🚨" if "SHORT" in msg_type else "🚀"
            header = f"🔥 <b>DEGEN SIGNAL: {msg_type}</b> 🔥"
            
            message = (
                f"{emoji} {header}\n\n"
                f"💵 <b>ENTRY PRICE: ${price:,.2f}</b>\n"
                f"💣 <b>LEVERAGE: 50x</b>\n"
                f"<code>=========================</code>\n"
                f"🎯 <b>TARGET:</b> ${tp:,.2f} (+{reward:.1f}%)\n"
                f"🛑 <b>STOP:</b>   ${sl:,.2f} (-{risk:.1f}%)\n"
                f"<code>=========================</code>\n\n"
                f"{wall_msg}\n"
                f"🧠 <b>WHY WE ARE APING:</b>\n"
                f"{reason}\n\n"
                f"🤖 <b>AI ROAST:</b>\n"
                f"<i>{ai_summary}</i>"
            )
        await bot.send_message(chat_id=TG_CHAT_ID, text=message, parse_mode='HTML')
    except Exception as e: print(f"⚠️ Telegram Error: {e}")

async def execute_avantis_trade(action_type, current_price, sl_price, tp_price):
    if SIMULATION_MODE: return "0xSimulatedHash"
    if not PRIVATE_KEY: return None
    try:
        client = TraderClient(BASE_RPC)
        client.set_local_signer(PRIVATE_KEY)
        trader_address = client.get_signer().address 
        is_long = "LONG" in action_type
        if "OPEN" in action_type:
            trade_input = TradeInput(
                trader=trader_address,
                pair_index=PAIR_INDEX,
                is_long=is_long,
                collateral_in_trade=TRADE_COLLATERAL,
                leverage=LEVERAGE,
                tp=tp_price,
                sl=sl_price,
                open_price=current_price
            )
            tx = await client.trade.open_market_trade(trade_input, TradeInputOrderType.MARKET, slippage=0.02)
            return tx.transaction_hash
        elif action_type == "CLOSE":
            positions = await client.trade.get_positions(trader_address)
            if positions: return (await client.trade.close_trade(positions[0].id)).transaction_hash
    except Exception as e: print(f"❌ Execution Error: {e}"); return None

async def main():
    print(f"🔥 Degen v9.8 (Walls in Telegram) | Mode: {'SIMULATION' if SIMULATION_MODE else 'REAL MONEY'}")

    old_state = load_state()
    is_in_position = old_state.get("is_open", False) if old_state else False
    entry_price = old_state.get("entry_price", 0) if old_state else 0
    active_signal_type = old_state.get("active_signal_type", None) if old_state else None
    signal_start_str = old_state.get("signal_start_time", None) if old_state else None
    signal_start_time = datetime.fromisoformat(signal_start_str) if signal_start_str else None

    current_win_prob = run_learning_cycle()
    last_learning = datetime.now()
    last_alert = datetime.min
    prev_oi = 0
    boot_time = datetime.now()

    signal_tp = 0
    signal_sl = 0

    status_freeze_until = None
    frozen_decision = ""
    frozen_reason = ""
    was_in_position = is_in_position

    while True:
        try:
            if (datetime.now() - boot_time).seconds < 60:
                fetch_smart_money_data(); await asyncio.sleep(5); continue

            if (datetime.now() - last_learning).seconds > 3600:
                print("🧠 RE-TRAINING AI MODEL (Hourly Cycle)...")
                current_win_prob = run_learning_cycle()
                last_learning = datetime.now()

            closes = fetch_candles()
            price = closes[-1] if len(closes) > 0 else 0
            fng = get_sentiment()
            slope, vol = calculate_quantum_wave(closes)
            whale_ratio, current_oi = fetch_smart_money_data()
            flow_state, oi_delta = analyze_flow_state(slope, current_oi, prev_oi)
            prev_oi = current_oi

            base_tp = price * (1 + DEFAULT_TP)
            base_sl = price * (1 - DEFAULT_SL)

            # FORCE WALL CHECK
            smart_tp, smart_sl, wall_info = adjust_smart_targets("LONG", price, base_tp, base_sl)
            raw_tp = smart_tp
            raw_sl = smart_sl

            # HOURLY ALERT (PASS WALL INFO)
            if (datetime.now() - last_alert).seconds > 3600:
                await send_telegram_alert(
                    "DEGEN HEARTBEAT", price, raw_tp, raw_sl, "Hourly Heartbeat", 
                    slope, whale_ratio, fng, flow_state, current_win_prob, walls=wall_info
                )
                last_alert = datetime.now()

            decision = "SCANNING"
            reason = f"Slope: {slope:.4f} | Whales: {whale_ratio:.2f}"
            pending_info = None

            if is_in_position:
                was_in_position = True
            elif was_in_position and not is_in_position:
                print("🎓 Trade Closed. Initiating IMMEDIATE Learning Cycle...")
                await asyncio.sleep(10)
                current_win_prob = run_learning_cycle()
                last_learning = datetime.now()
                was_in_position = False

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

            elif not is_in_position and not status_freeze_until:
                trend_up = slope > 0.5
                whale_buy = (whale_ratio > 1.2) and (fng < 40)
                detected = None
                if trend_up and (fng < 60 or whale_buy): detected = "LONG"
                elif slope < -0.5: detected = "SHORT"

                if detected:
                    if current_win_prob > 25:
                        signal_tp = smart_tp
                        signal_sl = smart_sl
                        adj_note = "Walls Respected"

                        if active_signal_type != detected:
                            active_signal_type = detected
                            signal_start_time = datetime.now()

                        decision = f"🚀 {detected} SIGNAL"
                        reason = f"High Velocity Setup + {adj_note}"
                        log_signal_to_history(detected, price, smart_sl, smart_tp, current_win_prob, reason)
                        
                        await send_telegram_alert(f"{detected} ETH", price, smart_tp, smart_sl, reason, slope, whale_ratio, fng, flow_state, current_win_prob, walls=wall_info)

            display_sl = signal_sl if active_signal_type else raw_sl
            display_tp = signal_tp if active_signal_type else raw_tp

            update_state("ACTIVE", is_in_position, entry_price, price, fng, slope, vol, current_win_prob, whale_ratio, decision, reason, display_sl, display_tp, flow_state, oi_delta, pending_info, active_signal_type, signal_start_time)
            await asyncio.sleep(60)

        except Exception as e: print(f"Loop Error: {e}"); await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
