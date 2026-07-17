"""
VWAP Reclaim CE — Backtest Rebuild
Nifty 50 Spot -> ATM CE (Black-Scholes simulated) | 2021-07-01 -> 2026-06-30

Rebuilds the backtest for VWAP Reclaim CE from its documented canonical spec
(indices-system/strategies/vwap_reclaim.md) -- original db/-era tooling retired
2026-07-10, no source file remaining. Historical record: 10 trades, 50% WR,
PF 2.95 (thin sample) -- this rebuild is a fresh, independently-reproducible
baseline.

Params (from vwap_reclaim.md):
  Daily trend filter: prev day close > 100-day EMA (daily closes)
  Anchored intraday VWAP (resets 9:15) | stretch >= 20 pts below VWAP required
  Reclaim: 15m candle closes back above VWAP | no volume filter
  Entry: open of candle after reclaim | cutoff 13:00 IST
  Stop: low of reclaim candle | Target: entry + 1.5x stretch_pts | hard exit 14:30
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

EMA_PERIOD = 100
STRETCH_MIN = 20
TARGET_RR = 1.5
ENTRY_CUTOFF = dtime(13, 0)
HARD_EXIT = dtime(14, 30)
LOT_SIZE = 65

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vwap_reclaim_rebuild")
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


print(f"Fetching daily Nifty data {START_DATE} -> {END_DATE} (trend filter) ...")
df_d = _fetch("D")
df_d["ema100"] = df_d["close"].ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean()
bull_by_date = (df_d["close"] > df_d["ema100"]).shift(1)  # prev day's close vs EMA, keyed to today
bull_by_date.index = df_d.index.date

print(f"Fetching 15m Nifty data {START_DATE} -> {END_DATE} (signal) ...")
df = _fetch("15m")
print(f"Loaded {len(df):,} bars: {df.index[0]} -> {df.index[-1]}")
df["date"] = df.index.date

trades = []
for day, grp in df.groupby("date"):
    if day not in bull_by_date.index or not bool(bull_by_date.get(day, False)):
        continue  # trend filter: prev day close must be above 100-day EMA

    session = grp.between_time("09:15", "15:30")
    if session.empty:
        continue

    typical = (session["high"] + session["low"] + session["close"]) / 3
    cum_pv = (typical * session["volume"]).cumsum()
    cum_vol = session["volume"].cumsum()
    vwap = cum_pv / cum_vol

    max_stretch = 0.0
    entered = False
    for i in range(len(session) - 1):
        if entered:
            break
        ts = session.index[i]
        if ts.time() > ENTRY_CUTOFF:
            break
        row = session.iloc[i]
        v = vwap.iloc[i]
        stretch_now = v - row["low"]
        if stretch_now > max_stretch:
            max_stretch = stretch_now

        reclaimed = max_stretch >= STRETCH_MIN and row["close"] > v
        if not reclaimed:
            continue

        entry_ts = session.index[i + 1]
        entry_bar = session.iloc[i + 1]
        entry_spot = entry_bar["open"]
        stop_spot = row["low"]
        target_spot = entry_spot + TARGET_RR * max_stretch

        expiry = nearest_weekly_expiry(day)
        strike = nearest_atm_strike(entry_spot)
        entry_premium = bs_price(entry_spot, strike, day, expiry, "CE")

        exit_spot, exit_ts, reason = None, None, None
        rest = session[session.index > entry_ts]
        for ts2, bar2 in rest.iterrows():
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
            "stretch_pts": round(max_stretch, 1), "stop_spot": stop_spot, "target_spot": round(target_spot, 1),
            "strike": strike, "expiry": expiry, "entry_premium": round(entry_premium, 2),
            "exit_time": exit_ts, "exit_spot": exit_spot, "exit_premium": round(exit_premium, 2),
            "pnl_pct": round(pnl_pct * 100, 2), "pnl_rupees": round(pnl_rupees, 2), "reason": reason,
        })
        entered = True

trades_df = pd.DataFrame(trades)
trades_df.to_csv(os.path.join(OUT_DIR, "vwap_reclaim_rebuild_trades.csv"), index=False)

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
    print(f"\n=== VWAP Reclaim CE Rebuild ===")
    print(f"Trades: {n} | WR: {wr:.1f}% | Avg P&L: {avg_pct:+.2f}% | PF: {pf:.2f}")
    print(f"Exit breakdown: {trades_df['reason'].value_counts().to_dict()}")
    print(f"\nHistorical record (unreproducible tooling, kept for reference): 10 trades, 50% WR, PF 2.95 (thin sample)")
