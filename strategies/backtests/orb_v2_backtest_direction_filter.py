"""
ORB_Spread — Direction-Aware Big-Trend Filter Test
Nifty 50 Spot | 2021-07-01 -> 2026-06-30

Follow-up to orb_v2_backtest_prevday_direction.py. That test showed the
continuation/fade split on the blocked big-trend bucket (prev_day_move >
0.42%) was weak (PF 1.08 vs 0.90), but splitting the SAME bucket by raw
signal direction was much cleaner: SHORT trades PF 1.16, LONG trades PF 0.84,
regardless of continuation/fade framing.

This script tests a direction-aware replacement for the current blanket
big-trend block:

  CURRENT RULE   : allow only if prev_day_move <= 0.42%
  CANDIDATE RULE : allow if prev_day_move <= 0.42%
                   OR (prev_day_move > 0.42% AND signal direction == SHORT)

i.e. keep blocking LONG signals on big-trend days, but let SHORT signals
(bear put spread) through.

Baseline config: HARD_EXIT 15:15, TARGET 40pts, STOP 25pts.
Prev-day threshold: 0.42% (the adopted live threshold).
"""

import os
import warnings
from datetime import time as dtime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -- Config -------------------------------------------------------------------
API_KEY = os.getenv("OPENALGO_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENALGO_API_KEY environment variable before running.")
HOST       = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
START_DATE = "2021-07-01"
END_DATE   = "2026-06-30"

OR_MIN, OR_MAX      = 30, 150
ENTRY_END           = dtime(12, 0)
HARD_EXIT           = dtime(15, 15)
TARGET_PTS, STOP_PTS = 40, 25
PREV_MOVE_THRESHOLD = 0.42   # adopted live threshold

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orb_v2_direction_filter")
os.makedirs(OUT_DIR, exist_ok=True)

# -- Fetch 5m data --------------------------------------------------------------
from openalgo import api as openalgo_api
import pytz

IST    = pytz.timezone("Asia/Kolkata")
client = openalgo_api(api_key=API_KEY, host=HOST)

resp = client.history(symbol="NIFTY", exchange="NSE_INDEX", interval="5m",
                      start_date=START_DATE, end_date=END_DATE)
if isinstance(resp, dict):
    if resp.get("status") != "success":
        raise SystemExit(f"API error: {resp}")
    df = pd.DataFrame(resp.get("data", []))
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
elif hasattr(resp, "empty"):
    df = resp
else:
    raise SystemExit(f"Unexpected response type: {type(resp)}")

df.columns = [c.lower() for c in df.columns]
df.index = df.index.tz_localize("Asia/Kolkata") if df.index.tz is None else df.index.tz_convert(IST)
df = df.sort_index()
print(f"Loaded {len(df):,} 5m bars: {df.index[0]} -> {df.index[-1]}")

# -- Daily stats — net move %, lagged by 1 day -------------------------------
daily = df.groupby(df.index.date).agg(open=("open", "first"), close=("close", "last"))
daily["net_move_pct"]  = (daily["close"] - daily["open"]).abs() / daily["open"] * 100
daily["prev_net_move"] = daily["net_move_pct"].shift(1)

# -- Signal generation (identical to adopted backtest) -----------------------
records = []
for day, grp in df.groupby(df.index.date):
    or_window = grp.between_time("09:15", "09:44")
    if len(or_window) < 3:
        continue
    orb_high, orb_low = or_window["high"].max(), or_window["low"].min()
    if not (OR_MIN <= orb_high - orb_low <= OR_MAX):
        continue

    at_1015 = grp.between_time("10:15", "10:15")
    if not at_1015.empty and orb_low < at_1015["close"].iloc[0] < orb_high:
        continue

    prev_net = daily["prev_net_move"].get(day, np.nan)

    sig = grp.between_time("09:45", "12:00")
    if len(sig) < 2:
        continue

    c, o, h, lo = sig["close"].values, sig["open"].values, sig["high"].values, sig["low"].values
    idx = sig.index
    bear_done = bull_done = False

    for i in range(1, len(sig)):
        if idx[i].time() > ENTRY_END:
            break
        if not bear_done and c[i-1] > orb_high and c[i] < orb_high and c[i] < o[i]:
            records.append({"day": day, "entry_time": idx[i], "direction": "SHORT",
                            "signal": "BearishReject", "entry_price": c[i], "prev_net": prev_net})
            bear_done = True
        if not bull_done and c[i-1] < orb_low and c[i] > orb_low and c[i] > o[i]:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                            "signal": "BullishReject", "entry_price": c[i], "prev_net": prev_net})
            bull_done = True
        if not bear_done and i >= 2 and h[i] < h[i-1] < h[i-2] and c[i] < orb_high:
            records.append({"day": day, "entry_time": idx[i], "direction": "SHORT",
                            "signal": "LowerHigh", "entry_price": c[i], "prev_net": prev_net})
            bear_done = True
        if not bull_done and i >= 2 and lo[i] > lo[i-1] > lo[i-2] and c[i] > orb_low:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                            "signal": "HigherLow", "entry_price": c[i], "prev_net": prev_net})
            bull_done = True

entries_df = pd.DataFrame(records).dropna(subset=["prev_net"]).reset_index(drop=True)
print(f"Total signals: {len(entries_df)}")


# -- Simulate exits -----------------------------------------------------------
def simulate(entries, df):
    out = entries.copy()
    out["exit_time"] = pd.NaT
    out["exit_price"] = np.nan
    out["pnl_pts"] = np.nan
    out["reason"] = ""

    for row_i, row in out.iterrows():
        entry_px = row["entry_price"]
        sign     = 1 if row["direction"] == "SHORT" else -1
        day_bars = df[df.index.date == row["day"]]

        for ts, bar in day_bars[day_bars.index >= row["entry_time"]].iterrows():
            if ts == row["entry_time"]:
                continue
            px  = bar["close"]
            pnl = (entry_px - px) * sign
            t   = ts.time()

            if pnl >= TARGET_PTS:
                reason = "TARGET"
            elif pnl <= -STOP_PTS:
                reason = "STOP"
            elif t >= HARD_EXIT:
                reason = "HARD_EXIT"
            else:
                continue

            out.at[row_i, "exit_time"]  = ts
            out.at[row_i, "exit_price"] = px
            out.at[row_i, "pnl_pts"]    = pnl
            out.at[row_i, "reason"]     = reason
            break

    return out.dropna(subset=["pnl_pts"]).copy()


def summarise(df_sub, label):
    if df_sub.empty:
        print(f"\n{label}: no trades")
        return {}
    n    = len(df_sub)
    wins = (df_sub["pnl_pts"] > 0).sum()
    wr   = wins / n * 100
    avg  = df_sub["pnl_pts"].mean()
    gw   = df_sub[df_sub["pnl_pts"] > 0]["pnl_pts"].sum()
    gl   = abs(df_sub[df_sub["pnl_pts"] <= 0]["pnl_pts"].sum())
    pf   = gw / gl if gl > 0 else float("inf")
    exits = df_sub["reason"].value_counts().to_dict()
    print(f"\n-- {label} " + "-" * max(1, 60 - len(label)))
    print(f"  Trades        : {n}")
    print(f"  Win rate      : {wr:.1f}%")
    print(f"  Avg P&L       : {avg:+.2f} pts")
    print(f"  Total P&L     : {df_sub['pnl_pts'].sum():+.1f} pts")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  Exits         : {exits}")
    return {"n": n, "wr": wr, "avg": avg, "total": df_sub["pnl_pts"].sum(), "pf": pf}


# -- Bucket definitions ---------------------------------------------------------
quiet    = entries_df[entries_df["prev_net"] <= PREV_MOVE_THRESHOLD]
bigtrend = entries_df[entries_df["prev_net"] >  PREV_MOVE_THRESHOLD]
bigtrend_short = bigtrend[bigtrend["direction"] == "SHORT"]
bigtrend_long  = bigtrend[bigtrend["direction"] == "LONG"]

current_rule_entries   = quiet.reset_index(drop=True)
candidate_rule_entries = pd.concat([quiet, bigtrend_short]).reset_index(drop=True)

all_trades       = simulate(entries_df, df)
current_trades   = simulate(current_rule_entries, df)
candidate_trades = simulate(candidate_rule_entries, df)
bigshort_trades  = simulate(bigtrend_short.reset_index(drop=True), df)
biglong_trades   = simulate(bigtrend_long.reset_index(drop=True), df)

all_trades.to_csv(os.path.join(OUT_DIR, "all_trades.csv"), index=False)
current_trades.to_csv(os.path.join(OUT_DIR, "current_rule_trades.csv"), index=False)
candidate_trades.to_csv(os.path.join(OUT_DIR, "candidate_rule_trades.csv"), index=False)

print("\n" + "=" * 70)
print("COMPARISON: CURRENT RULE vs CANDIDATE DIRECTION-AWARE RULE")
print("=" * 70)
summarise(all_trades,       "Unfiltered (all signals, no prev-day filter)")
summarise(current_trades,   f"CURRENT RULE: quiet only (prev_net <= {PREV_MOVE_THRESHOLD}%)")
summarise(candidate_trades, f"CANDIDATE RULE: quiet OR big-trend SHORT")
summarise(bigshort_trades,  "  (component) big-trend SHORT added by candidate rule")
summarise(biglong_trades,   "  (reference) big-trend LONG, stays blocked under both rules")

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
rows = [
    ("Unfiltered", all_trades),
    ("Current rule (quiet only)", current_trades),
    ("Candidate rule (quiet + big-trend SHORT)", candidate_trades),
]
print(f"  {'Rule':<42} {'N':>6} {'WR%':>7} {'Avg':>8} {'PF':>7}")
print("  " + "-" * 70)
for label, trades in rows:
    if trades.empty:
        print(f"  {label:<42} {'-':>6}")
        continue
    n  = len(trades)
    wr = (trades["pnl_pts"] > 0).mean() * 100
    avg = trades["pnl_pts"].mean()
    gw = trades[trades["pnl_pts"] > 0]["pnl_pts"].sum()
    gl = abs(trades[trades["pnl_pts"] <= 0]["pnl_pts"].sum())
    pf = gw / gl if gl > 0 else float("inf")
    print(f"  {label:<42} {n:>6} {wr:>7.1f} {avg:>8.2f} {pf:>7.2f}")

print("\nCSVs saved to:", OUT_DIR)
print("Backtest complete.")