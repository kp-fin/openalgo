"""
PDH Breakout CE — Backtest Rebuild
Nifty 50 Spot -> ATM CE (Black-Scholes simulated) | 2021-07-01 -> 2026-06-30

Rebuilds the backtest for PDH Breakout CE from its documented canonical spec
(indices-system/strategies/pdh_breakout.md) -- original db/-era tooling retired
2026-07-10, no source file remaining. Historical record: 464 trades, 44.8% WR,
PF 1.24 -- this rebuild is a fresh, independently-reproducible baseline.

Params (from pdh_breakout.md):
  PDH = max(high) of previous session (9:15-15:30)
  Breakout: 15m candle CLOSES above PDH | volume > 1.3x 20-candle rolling avg
  Entry: open of candle after breakout close | window 9:15-11:15 only
  Stop: PDH - 10 pts | Target: entry + 1.5x risk | hard exit 14:30 IST
  One trade/day, first qualifying breakout only
  CE, ATM strike (nearest 50), nearest Thursday expiry, BS pricing (IV=15%)
"""

import os
import sys
import warnings
from datetime import time as dtime

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bs_pricing import nearest_atm_strike, nearest_weekly_expiry, bs_price

API_KEY = os.getenv("OPENALGO_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENALGO_API_KEY environment variable before running.")
HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
START_DATE = "2021-07-01"
END_DATE = "2026-06-30"
IST = pytz.timezone("Asia/Kolkata")

VOL_MULT = 1.3
VOL_LOOKBACK = 20
STOP_BUFFER = 10
TARGET_RR = 1.5
ENTRY_WINDOW_END = dtime(11, 15)
HARD_EXIT = dtime(14, 30)
LOT_SIZE = 65

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdh_breakout_rebuild")
os.makedirs(OUT_DIR, exist_ok=True)

from openalgo import api as openalgo_api
client = openalgo_api(api_key=API_KEY, host=HOST)


def _fetch(interval):
    resp = client.history(symbol="NIFTY", exchange="NSE_INDEX", interval=interval,
                           start_date=START_DATE, end_date=END_DATE)
    if isinstance(resp, dict):
        if resp.get("status") != "success":
            raise RuntimeError(f"API error ({interval}): {resp}")
        out = pd.DataFrame(resp.get("data", []))
        out["datetime"] = pd.to_datetime(out["datetime"])
        out = out.set_index("datetime")
    else:
        out = resp
    out.columns = [c.lower() for c in out.columns]
    if out.index.tz is None:
        out.index = out.index.tz_localize("Asia/Kolkata")
    else:
        out.index = out.index.tz_convert(IST)
    return out.sort_index()


print(f"Fetching 15m Nifty data {START_DATE} -> {END_DATE} ...")
df = _fetch("15m")
print(f"Loaded {len(df):,} bars: {df.index[0]} -> {df.index[-1]}")

df["date"] = df.index.date
daily_high = df.groupby("date")["high"].max()
pdh_by_date = daily_high.shift(1)  # previous session's high, keyed by trading date
df["pdh"] = df["date"].map(pdh_by_date)
df["vol_avg"] = df["volume"].rolling(VOL_LOOKBACK).mean()

trades = []
traded_dates = set()
for day, grp in df.groupby("date"):
    grp = grp.dropna(subset=["pdh", "vol_avg"])
    if grp.empty:
        continue
    window = grp.between_time("09:15", ENTRY_WINDOW_END.strftime("%H:%M"))
    entered = False
    for i in range(len(window) - 1):
        row = window.iloc[i]
        ts = window.index[i]
        if entered:
            break
        signal = row["close"] > row["pdh"] and row["volume"] > VOL_MULT * row["vol_avg"]
        if not signal:
            continue

        # entry at open of next candle (may be outside `window` if breakout is the last window bar)
        full_day = df[df["date"] == day]
        pos = full_day.index.get_loc(ts)
        if pos + 1 >= len(full_day):
            continue
        entry_ts = full_day.index[pos + 1]
        entry_bar = full_day.iloc[pos + 1]

        pdh = row["pdh"]
        entry_spot = entry_bar["open"]
        stop_spot = pdh - STOP_BUFFER
        risk = entry_spot - stop_spot
        if risk <= 0:
            continue  # degenerate case: entry already below stop level
        target_spot = entry_spot + TARGET_RR * risk

        expiry = nearest_weekly_expiry(day)
        strike = nearest_atm_strike(entry_spot)
        entry_premium = bs_price(entry_spot, strike, day, expiry, "CE")

        exit_spot, exit_ts, reason = None, None, None
        rest_of_day = full_day[full_day.index > entry_ts]
        for ts2, bar2 in rest_of_day.iterrows():
            t = ts2.time()
            if bar2["close"] <= stop_spot:
                exit_spot, exit_ts, reason = bar2["close"], ts2, "STOP"
                break
            if bar2["close"] >= target_spot:
                exit_spot, exit_ts, reason = bar2["close"], ts2, "TARGET"
                break
            if t >= HARD_EXIT:
                exit_spot, exit_ts, reason = bar2["close"], ts2, "HARD_EXIT"
                break
        if exit_spot is None:
            continue

        exit_premium = bs_price(exit_spot, strike, day, expiry, "CE")
        pnl_pct = (exit_premium - entry_premium) / entry_premium
        pnl_rupees = (exit_premium - entry_premium) * LOT_SIZE
        trades.append({
            "day": day, "signal_time": ts, "entry_time": entry_ts, "entry_spot": entry_spot,
            "pdh": pdh, "stop_spot": stop_spot, "target_spot": target_spot, "strike": strike,
            "expiry": expiry, "entry_premium": round(entry_premium, 2),
            "exit_time": exit_ts, "exit_spot": exit_spot, "exit_premium": round(exit_premium, 2),
            "pnl_pct": round(pnl_pct * 100, 2), "pnl_rupees": round(pnl_rupees, 2), "reason": reason,
        })
        entered = True

trades_df = pd.DataFrame(trades)
trades_df.to_csv(os.path.join(OUT_DIR, "pdh_breakout_rebuild_trades.csv"), index=False)

if trades_df.empty:
    print("No trades generated.")
else:
    n = len(trades_df)
    wins = (trades_df["pnl_pct"] > 0).sum()
    wr = wins / n * 100
    avg_pct = trades_df["pnl_pct"].mean()
    gw = trades_df[trades_df["pnl_pct"] > 0]["pnl_pct"].sum()
    gl = abs(trades_df[trades_df["pnl_pct"] <= 0]["pnl_pct"].sum())
    pf = gw / gl if gl > 0 else float("inf")
    print(f"\n=== PDH Breakout CE Rebuild ===")
    print(f"Trades: {n} | WR: {wr:.1f}% | Avg P&L: {avg_pct:+.2f}% | PF: {pf:.2f}")
    print(f"Exit breakdown: {trades_df['reason'].value_counts().to_dict()}")
    print(f"\nHistorical record (unreproducible tooling, kept for reference): 464 trades, 44.8% WR, PF 1.24")
