"""
ORB v2 — Peachy Rejection Method — VectorBT Backtest
Nifty 50 Spot | 5m candles | 2021-07-01 → 2026-06-30

Opening Range: 9:15–9:45 IST (30m)
Signals (after 9:45, before 12:00):
  Bearish Reject : close[1] > orb_high AND close < orb_high AND bearish candle  → proxy SHORT (PE)
  Bullish Reject : close[1] < orb_low  AND close < orb_low  AND bullish candle  → proxy LONG  (CE)
  Lower High     : 3 consecutive lower highs below orb_high                     → proxy SHORT (PE)
  Higher Low     : 3 consecutive higher lows above orb_low                      → proxy LONG  (CE)
  Range day skip : spot still inside OR at 10:15 IST → no trades that day

P&L in spot points (proxy). Actual option premium P&L tracked in OpenAlgo Sandbox.
Target: +40 pts | Stop: -25 pts | Hard exit: 14:30 IST
"""

import os
import json
import warnings
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import quantstats as qs

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY    = os.getenv("OPENALGO_API_KEY", "your_openalgo_api_key_here")
HOST       = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
START_DATE = "2021-07-01"
END_DATE   = "2026-06-30"

OR_MIN     = 30    # skip day if 30m OR range below this (pts)
OR_MAX     = 150   # skip day if 30m OR range above this (pts)
TARGET_PTS = 40    # spot pts target (~+40% of 100pt ATM premium proxy)
STOP_PTS   = 25    # spot pts stop  (~-25% of 100pt ATM premium proxy)
ENTRY_END  = dtime(12, 0)
HARD_EXIT  = dtime(14, 30)
RANGE_CHK  = dtime(10, 15)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orb_v2")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Fetch data ────────────────────────────────────────────────────────────────
try:
    from openalgo import api as openalgo_api
    import pytz

    IST    = pytz.timezone("Asia/Kolkata")
    client = openalgo_api(api_key=API_KEY, host=HOST)

    print(f"Fetching 5m Nifty data {START_DATE} → {END_DATE} ...")
    df = client.history(
        symbol="NIFTY",
        exchange="NSE_INDEX",
        interval="5m",
        start_date=START_DATE,
        end_date=END_DATE,
    )
    if df is None or df.empty:
        raise RuntimeError("Empty response from OpenAlgo. Check API key and connection.")

    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)
    df = df.sort_index()
    df.columns = [c.lower() for c in df.columns]
    print(f"Loaded {len(df):,} bars: {df.index[0]} → {df.index[-1]}")

except Exception as exc:
    raise SystemExit(f"Data fetch failed: {exc}")

# ── Signal generation ─────────────────────────────────────────────────────────
records = []  # one row per trade

total_days      = 0
skip_range      = 0
skip_range_day  = 0

for day, grp in df.groupby(df.index.date):
    total_days += 1

    # 30m opening range 9:15–9:44
    or_window = grp.between_time("09:15", "09:44")
    if len(or_window) < 3:
        continue
    orb_high = or_window["high"].max()
    orb_low  = or_window["low"].min()
    or_range = orb_high - orb_low

    if or_range < OR_MIN or or_range > OR_MAX:
        skip_range += 1
        continue

    # Range day filter: still inside OR at 10:15?
    at_1015 = grp.between_time("10:15", "10:15")
    if not at_1015.empty:
        c1015 = at_1015["close"].iloc[0]
        if orb_low < c1015 < orb_high:
            skip_range_day += 1
            continue

    # Signal window 9:45–12:00
    sig = grp.between_time("09:45", "12:00")
    if len(sig) < 2:
        continue

    c = sig["close"].values
    o = sig["open"].values
    h = sig["high"].values
    lo = sig["low"].values
    idx = sig.index

    bear_done = bull_done = False

    for i in range(1, len(sig)):
        t = idx[i].time()
        if t > ENTRY_END:
            break

        # ── Primary: Bearish Reject → SHORT proxy ─────────────────────────
        if not bear_done:
            if c[i-1] > orb_high and c[i] < orb_high and c[i] < o[i]:
                records.append({"day": day, "entry_time": idx[i], "direction": "SHORT",
                                 "signal": "BearishReject", "entry_price": c[i],
                                 "orb_high": orb_high, "orb_low": orb_low})
                bear_done = True

        # ── Primary: Bullish Reject → LONG proxy ──────────────────────────
        if not bull_done:
            if c[i-1] < orb_low and c[i] > orb_low and c[i] > o[i]:
                records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                                 "signal": "BullishReject", "entry_price": c[i],
                                 "orb_high": orb_high, "orb_low": orb_low})
                bull_done = True

        # ── Confirmation: Lower High → SHORT proxy ─────────────────────────
        if not bear_done and i >= 2:
            if h[i] < h[i-1] < h[i-2] and c[i] < orb_high:
                records.append({"day": day, "entry_time": idx[i], "direction": "SHORT",
                                 "signal": "LowerHigh", "entry_price": c[i],
                                 "orb_high": orb_high, "orb_low": orb_low})
                bear_done = True

        # ── Confirmation: Higher Low → LONG proxy ─────────────────────────
        if not bull_done and i >= 2:
            if lo[i] > lo[i-1] > lo[i-2] and c[i] > orb_low:
                records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                                 "signal": "HigherLow", "entry_price": c[i],
                                 "orb_high": orb_high, "orb_low": orb_low})
                bull_done = True

print(f"Days: {total_days} total | {skip_range} skipped (range) | {skip_range_day} skipped (range day)")
print(f"Signals generated: {len(records)}")

if not records:
    raise SystemExit("No signals generated — check data or parameters.")

# ── Simulate exits ────────────────────────────────────────────────────────────
entries_df = pd.DataFrame(records)
entries_df["exit_time"]  = pd.NaT
entries_df["exit_price"] = np.nan
entries_df["pnl_pts"]    = np.nan
entries_df["reason"]     = ""

for row_i, row in entries_df.iterrows():
    entry_px = row["entry_price"]
    sign     = 1 if row["direction"] == "SHORT" else -1  # SHORT: profit when price falls
    day_bars = df[df.index.date == row["day"]]

    for ts, bar in day_bars[day_bars.index >= row["entry_time"]].iterrows():
        if ts == row["entry_time"]:
            continue
        px     = bar["close"]
        pnl    = (entry_px - px) * sign   # SHORT profits when price drops
        t      = ts.time()

        if pnl >= TARGET_PTS:
            reason = "TARGET"
        elif pnl <= -STOP_PTS:
            reason = "STOP"
        elif t >= HARD_EXIT:
            reason = "HARD_EXIT"
        else:
            continue

        entries_df.at[row_i, "exit_time"]  = ts
        entries_df.at[row_i, "exit_price"] = px
        entries_df.at[row_i, "pnl_pts"]    = pnl
        entries_df.at[row_i, "reason"]     = reason
        break

trades = entries_df.dropna(subset=["pnl_pts"]).copy()

# ── Results ───────────────────────────────────────────────────────────────────
def summarise(df_sub, label):
    if df_sub.empty:
        print(f"\n{label}: no trades")
        return
    n      = len(df_sub)
    wins   = (df_sub["pnl_pts"] > 0).sum()
    wr     = wins / n * 100
    avg    = df_sub["pnl_pts"].mean()
    gw     = df_sub[df_sub["pnl_pts"] > 0]["pnl_pts"].sum()
    gl     = abs(df_sub[df_sub["pnl_pts"] <= 0]["pnl_pts"].sum())
    pf     = gw / gl if gl > 0 else float("inf")
    exits  = df_sub["reason"].value_counts().to_dict()
    sigs   = df_sub["signal"].value_counts().to_dict()
    print(f"\n── {label} ─────────────────────────────────────")
    print(f"  Trades        : {n}")
    print(f"  Win rate      : {wr:.1f}%")
    print(f"  Avg P&L       : {avg:+.2f} pts")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  Exits         : {exits}")
    print(f"  Signal types  : {sigs}")

summarise(trades[trades["direction"] == "SHORT"], "BEARISH (PE proxy)")
summarise(trades[trades["direction"] == "LONG"],  "BULLISH (CE proxy)")
summarise(trades, "ALL TRADES")

# ── Save outputs ──────────────────────────────────────────────────────────────
trades_path = os.path.join(OUT_DIR, "orb_v2_trades.csv")
trades.to_csv(trades_path, index=False)
print(f"\nTrades saved: {trades_path}")

# QuantStats tearsheet (equity curve in pts)
try:
    eq = trades.set_index("exit_time")["pnl_pts"].sort_index()
    eq.index = pd.to_datetime(eq.index).tz_localize(None)
    qs.reports.html(eq, output=os.path.join(OUT_DIR, "orb_v2_tearsheet.html"),
                    title="ORB v2 — Peachy Rejection Backtest")
    print("Tearsheet saved.")
except Exception as e:
    print(f"Tearsheet skipped: {e}")

print("\nBacktest complete.")
