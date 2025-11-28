import sqlite3
import pandas as pd
from datetime import datetime

DB_FILE = "data/quantum.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Table to store raw market snapshots for training
    c.execute('''CREATE TABLE IF NOT EXISTS snapshots
                 (timestamp TEXT, price REAL, sentiment INTEGER, 
                  slope REAL, volatility REAL, outcome REAL)''')
    conn.commit()
    conn.close()

def log_snapshot(price, sentiment, slope, volatility):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    ts = datetime.now().isoformat()
    c.execute("INSERT INTO snapshots (timestamp, price, sentiment, slope, volatility, outcome) VALUES (?, ?, ?, ?, ?, ?)",
              (ts, price, sentiment, slope, volatility, 0)) # Outcome 0 for now
    conn.commit()
    conn.close()

def get_recent_data(hours=168): # Default 7 days
    conn = sqlite3.connect(DB_FILE)
    # Get recent candles to simulate against
    # Note: In a real ML rig, we'd query our own DB. 
    # For this v1, we will just pull fresh Binance data in the learner to keep it fast.
    return None
