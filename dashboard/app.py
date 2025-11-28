import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import savgol_filter
import requests
import json
import os
import time
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="Quantum Command", layout="wide", page_icon="⚔️")

# Auto-refresh every 5 seconds
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > 5:
    st.session_state.last_refresh = time.time()
    st.rerun()

# --- CSS STYLING ---
st.markdown("""
<style>
    /* --- ANIMATIONS --- */
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    @keyframes breathe {
        0% { opacity: 0.6; }
        50% { opacity: 1.0; }
        100% { opacity: 0.6; }
    }

    /* --- CARDS --- */
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #333;
    }

    /* --- STATUS BADGES --- */
    .status-heartbeat {
        background-color: #1E293B;
        color: #94A3B8;
        border: 1px solid #334155;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        font-family: monospace;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .heartbeat-dot {
        height: 10px;
        width: 10px;
        background-color: #3b82f6;
        border-radius: 50%;
        display: inline-block;
        animation: breathe 2s infinite ease-in-out;
    }

    .status-long {
        background-color: #064E3B;
        color: #6EE7B7;
        border: 2px solid #10B981;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: 900;
        font-size: 24px;
        animation: pulse-green 2s infinite;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .status-short {
        background-color: #450A0A;
        color: #FCA5A5;
        border: 2px solid #EF4444;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: 900;
        font-size: 24px;
        animation: pulse-red 2s infinite;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .status-invalid {
        background-color: #374151;
        color: #9CA3AF;
        border: 1px dashed #6B7280;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-style: italic;
    }

    .header-safe { color: #3b82f6; border-bottom: 2px solid #3b82f6; margin-bottom: 10px; }
    .header-degen { color: #a855f7; border-bottom: 2px solid #a855f7; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- LOADERS ---
def load_json(filename):
    if os.path.exists(filename):
        try: 
            with open(filename, 'r') as f: return json.load(f)
        except: pass
    return {"decision": "OFFLINE", "leverage": 0, "win_probability": 0}

def load_csv(filename):
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            if 'Timestamp' in df.columns: df = df.sort_values(by='Timestamp', ascending=False)
            return df
        except: pass
    return pd.DataFrame()

def calculate_fib_levels(high, low):
    diff = high - low
    return {
        "0.5": high - (0.5 * diff),
        "0.618": high - (0.618 * diff),
        "0.786": high - (0.786 * diff)
    }

# Load State
safe_state = load_json("data/bot_state.json")
degen_state = load_json("data/degen_state.json")
safe_history = load_csv("data/signals.csv")
degen_history = load_csv("data/degen_signals.csv")

st.title("⚔️ Quantum Command: Twin Engines")

# --- RENDER CARD FUNCTION ---
def render_engine_card(title, state, css_header):
    with st.container():
        st.markdown(f"<h3 class='{css_header}'>{title}</h3>", unsafe_allow_html=True)
        
        decision = state.get("decision", "OFFLINE").upper()
        price = state.get("current_price", 0)
        entry = state.get("entry_price", 0)
        sl = state.get("suggested_sl", 0)
        tp = state.get("suggested_tp", 0)
        win_prob = state.get("win_probability", 0)
        reason = state.get("reason", "Initializing...")

        # --- VISUAL STATUS BADGE ---
        if "SIGNAL" in decision and "INVALID" not in decision:
            if "LONG" in decision:
                st.markdown(f"<div class='status-long'>🚀 LONG SIGNAL DETECTED</div>", unsafe_allow_html=True)
            elif "SHORT" in decision:
                st.markdown(f"<div class='status-short'>📉 SHORT SIGNAL DETECTED</div>", unsafe_allow_html=True)
            st.warning("⚠️ **MARKET ENTRY OPPORTUNITY**")
        
        elif "HOLDING" in decision:
            st.markdown(f"<div class='status-long' style='animation: none; border: 2px solid gold; color: gold;'>💰 TRADE ACTIVE (HOLDING)</div>", unsafe_allow_html=True)

        elif "INVALID" in decision or "CANCEL" in decision:
            st.markdown(f"<div class='status-invalid'>🚫 {decision}</div>", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class='status-heartbeat'>
                <span class='heartbeat-dot'></span> SYSTEM NOMINAL | SCANNING
            </div>
            """, unsafe_allow_html=True)

        # --- DYNAMIC METRICS GRID ---
        # If we are in a Signal or Trade, we need 4 columns to show Entry vs Current
        if "SIGNAL" in decision or "HOLDING" in decision:
            c1, c2, c3, c4 = st.columns(4)
            
            # 1. ENTRY PRICE
            c1.metric("Entry Price", f"${entry:,.2f}")
            
            # 2. CURRENT PRICE (With PnL Delta)
            delta = 0
            if entry > 0:
                raw_delta = (price - entry) / entry * 100
                if "SHORT" in decision: raw_delta = -raw_delta # Flip for short
                delta = f"{raw_delta:.2f}%"
            
            c2.metric("Current Price", f"${price:,.2f}", delta=delta)
            
            # 3. TP & SL
            c3.metric("Target (TP)", f"${tp:,.2f}")
            c4.metric("Stop (SL)", f"${sl:,.2f}")

        else:
            # Heartbeat Mode (3 Columns) - Entry is irrelevant
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"${price:,.2f}")
            c2.metric("Hypothetical TP", f"${tp:,.2f}")
            c3.metric("Hypothetical SL", f"${sl:,.2f}")

        # AI Confidence
        if win_prob > 0:
            st.write(f"**AI Confidence:** {win_prob:.1f}%")
            st.progress(int(min(win_prob, 100)))
        
        st.caption(f"📝 **Logic:** {reason}")

col1, col2 = st.columns(2)
with col1: render_engine_card("🛡️ SAFE ENGINE (5x)", safe_state, "header-safe")
with col2: render_engine_card("🔥 DEGEN ENGINE (50x)", degen_state, "header-degen")

st.markdown("---")

# --- CHART ---
st.subheader("📊 Live Market Structure")
def fetch_market_data():
    try:
        url = "https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=4h&limit=200"
        res = requests.get(url, timeout=2).json()
        df = pd.DataFrame(res, columns=['time', 'open', 'high', 'low', 'close', 'vol', 'x', 'y', 'z', 'a', 'b', 'c'])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return df
    except: return pd.DataFrame()

df = fetch_market_data()
if not df.empty:
    df['smooth'] = savgol_filter(df['close'], 21, 3)
    local_high = df['high'].max()
    local_low = df['low'].min()
    fibs = calculate_fib_levels(local_high, local_low)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['close'], mode='lines', line=dict(color='#606060', width=1), name='Price'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['smooth'], mode='lines', line=dict(color='#00FF99', width=2), name='Quantum Wave'))

    # Fibs
    fig.add_hline(y=fibs['0.618'], line_dash="dot", line_color="gold", opacity=0.5, annotation_text="Fib 0.618")
    fig.add_hline(y=fibs['0.786'], line_dash="dot", line_color="orange", opacity=0.5, annotation_text="Fib 0.786")

    # Entries (Visual Lines)
    if safe_state.get("is_open"):
        fig.add_hline(y=safe_state.get("entry_price",0), line_dash="dash", line_color="#3b82f6", annotation_text="SAFE ENTRY")
    if degen_state.get("is_open"):
        fig.add_hline(y=degen_state.get("entry_price",0), line_dash="dot", line_color="#a855f7", annotation_text="DEGEN ENTRY")

    fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- HISTORY ---
tab1, tab2 = st.tabs(["🛡️ Safe Logs", "🔥 Degen Logs"])
with tab1: st.dataframe(safe_history, use_container_width=True, hide_index=True)
with tab2: st.dataframe(degen_history, use_container_width=True, hide_index=True)
