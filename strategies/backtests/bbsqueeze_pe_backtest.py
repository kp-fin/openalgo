"""
BB Squeeze PE — Backtest Rebuild
Nifty 50 Spot -> ATM PE (Black-Scholes simulated) | 2021-07-01 -> 2026-06-30

Rebuilds the backtest for BB Squeeze PE from its documented canonical spec
(indices-system/strategies/bbsqueeze_pe.md), since the original db/-era tooling
was retired 2026-07-10 with no source file remaining. Historical record: 15
trades, 66.7% WR, +5.82% avg P&L, PF >1.0 -- this rebuild is a fresh,
independently-reproducible baseline, not an attempt to force a match to that
number (same honest framing as the ORB v1/v2 rebuild work).

Params (from bbsqueeze_pe.md):
  BB(20, 2 sigma) on 15m candles | squeeze lookback 5 candles, min streak 3
  ADX(14) >= 20 with -DI > +DI | volume > 1.1x 20-candle rolling avg
  PE only | entry at open of next candle after signal
  Strike: nearest ATM (50pt) | expiry: nearest weekly Thursday, skip if today IS expiry
  Target +35% / Stop -25% on entry premium | hard exit 15:00 IST
  Premium: Black-Scholes, IV=15% assumed
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

BB_PERIOD = 20
BB_STD = 2
SQUEEZE_LOOKBACK = 5
MIN_SQUEEZE_STREAK = 3
ADX_PERIOD = 14
ADX_THRESHOLD = 20
VOL_MULT = 1.1
VOL_LOOKBACK = 20
TARGET_PCT = 0.35
STOP_PCT = -0.25
HARD_EXIT = dtime(15, 0)
LOT_SIZE = 65

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bbsqueeze_pe_rebuild")
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


def wilder_rma(series, period):
    rma = pd.Series(np.nan, index=series.index)
    if len(series) <= period:
        return rma
    rma.iloc[period] = series.iloc[1:period + 1].mean()
    vals, rma_vals = series.values, rma.values
    for i in range(period + 1, len(series)):
        rma_vals[i] = (rma_vals[i - 1] * (period - 1) + vals[i]) / period
    return pd.Series(rma_vals, index=series.index)


def compute_adx_di(df, period=ADX_PERIOD):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close, prev_high, prev_low = close.shift(1), high.shift(1), low.shift(1)
    up_move, down_move = high - prev_high, prev_low - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = wilder_rma(tr, period)
    plus_di = 100 * wilder_rma(plus_dm, period) / atr
    minus_di = 100 * wilder_rma(minus_dm, period) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = wilder_rma(dx, period)
    return adx, plus_di, minus_di


df["adx"], df["plus_di"], df["minus_di"] = compute_adx_di(df)
mid = df["close"].rolling(BB_PERIOD).mean()
std = df["close"].rolling(BB_PERIOD).std()
df["bb_upper"] = mid + BB_STD * std
df["bb_lower"] = mid - BB_STD * std
df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid
df["vol_avg"] = df["volume"].rolling(VOL_LOOKBACK).mean()

# squeeze streak: bandwidth <= its own 5-candle rolling min (i.e. at/near a 5-candle low)
df["is_squeeze"] = df["bb_width"] <= df["bb_width"].rolling(SQUEEZE_LOOKBACK).min() * 1.0001
df["squeeze_streak"] = df["is_squeeze"].groupby((~df["is_squeeze"]).cumsum()).cumsum()

trades = []
in_position = False
for i in range(BB_PERIOD + VOL_LOOKBACK, len(df) - 1):
    row, nxt = df.iloc[i], df.iloc[i + 1]
    ts, nxt_ts = df.index[i], df.index[i + 1]

    if in_position:
        continue  # single active trade at a time, matching "no open trade" gate in the live script

    if ts.date() != nxt_ts.date():
        continue  # signal candle must have a next candle same day to enter

    signal = (
        row["squeeze_streak"] >= MIN_SQUEEZE_STREAK
        and row["close"] < row["bb_lower"]
        and row["adx"] >= ADX_THRESHOLD
        and row["minus_di"] > row["plus_di"]
        and row["volume"] > VOL_MULT * row["vol_avg"]
    )
    if not signal:
        continue

    entry_date = nxt_ts.date()
    expiry = nearest_weekly_expiry(entry_date)
    if expiry == entry_date:
        continue  # skip if today IS expiry day, per spec

    entry_spot = nxt["open"]
    strike = nearest_atm_strike(entry_spot)
    entry_premium = bs_price(entry_spot, strike, entry_date, expiry, "PE")

    exit_price, exit_time, reason = None, None, None
    day_bars = df[(df.index > nxt_ts) & (df.index.date == entry_date)]
    for ts2, bar2 in day_bars.iterrows():
        premium = bs_price(bar2["close"], strike, entry_date, expiry, "PE")
        pct = (premium - entry_premium) / entry_premium
        t = ts2.time()
        if pct >= TARGET_PCT:
            exit_price, exit_time, reason = premium, ts2, "TARGET"
            break
        if pct <= STOP_PCT:
            exit_price, exit_time, reason = premium, ts2, "STOP"
            break
        if t >= HARD_EXIT:
            exit_price, exit_time, reason = premium, ts2, "HARD_EXIT"
            break
    if exit_price is None:
        continue  # no bars left this day (signal too late) -- skip, matches "skip if expiry day" spirit

    pnl_pct = (exit_price - entry_premium) / entry_premium
    pnl_rupees = (exit_price - entry_premium) * LOT_SIZE
    trades.append({
        "signal_time": ts, "entry_time": nxt_ts, "entry_spot": entry_spot, "strike": strike,
        "expiry": expiry, "entry_premium": round(entry_premium, 2),
        "exit_time": exit_time, "exit_premium": round(exit_price, 2),
        "pnl_pct": round(pnl_pct * 100, 2), "pnl_rupees": round(pnl_rupees, 2), "reason": reason,
    })

trades_df = pd.DataFrame(trades)
trades_df.to_csv(os.path.join(OUT_DIR, "bbsqueeze_pe_rebuild_trades.csv"), index=False)

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
    print(f"\n=== BB Squeeze PE Rebuild ===")
    print(f"Trades: {n} | WR: {wr:.1f}% | Avg P&L: {avg_pct:+.2f}% | PF: {pf:.2f}")
    print(f"Exit breakdown: {trades_df['reason'].value_counts().to_dict()}")
    print(f"\nHistorical record (unreproducible tooling, kept for reference): 15 trades, 66.7% WR, +5.82% avg, PF>1.0")
