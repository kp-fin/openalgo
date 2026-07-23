"""
ORB_Spread — Previous-Day Direction Filter Test
Nifty 50 Spot | 2021-07-01 → 2026-06-30

Hypothesis: within the big-trend bucket (prev_day_move > 0.42%, currently
blocked), does signal direction relative to yesterday's move matter?

  CONTINUATION: signal direction matches yesterday's trend
    - Yesterday bearish (close < open) + today SHORT (bear put spread)
    - Yesterday bullish (close > open) + today LONG  (bull call spread)

  FADE: signal direction opposes yesterday's trend
    - Yesterday bearish + today LONG
    - Yesterday bullish + today SHORT

If continuation trades within the big-trend bucket have positive PF,
the filter could be loosened to allow those specifically.

Baseline config: HARD_EXIT 15:15, TARGET 40pts, STOP 25pts.
Prev-day threshold: 0.42% (the adopted live threshold).
"""

import os
import warnings
from datetime import time as dtime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("OPENALGO_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENALGO_API_KEY environment variable before running.")
HOST       = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
START_DATE = "2021-07-01"
END_DATE   = "2026-06-30"

OR_MIN, OR_MAX      = 30, 150
ENTRY_END           = dtime(12, 0)
HARD_EXIT           = dtime(15, 15)
RANGE_CHK           = dtime(10, 15)
TARGET_PTS, STOP_PTS = 40, 25
PREV_MOVE_THRESHOLD = 0.42   # adopted live threshold

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orb_v2_prevday_direction")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Fetch 5m data ─────────────────────────────────────────────────────────────
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
print(f"Loaded {len(df):,} 5m bars: {df.index[0]} → {df.index[-1]}")

# ── Daily stats — net move % and direction, lagged by 1 day ──────────────────
daily = df.groupby(df.index.date).agg(open=("open", "first"), close=("close", "last"))
daily["net_move_pct"]  = (daily["close"] - daily["open"]).abs() / daily["open"] * 100
daily["raw_move_pct"]  = (daily["close"] - daily["open"]) / daily["open"] * 100  # signed
daily["prev_net_move"] = daily["net_move_pct"].shift(1)
daily["prev_raw_move"] = daily["raw_move_pct"].shift(1)   # positive = yesterday bullish, negative = bearish

# ── Signal generation (identical to adopted backtest) ─────────────────────────
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

    prev_net  = daily["prev_net_move"].get(day, np.nan)
    prev_raw  = daily["prev_raw_move"].get(day, np.nan)

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
                            "signal": "BearishReject", "entry_price": c[i],
                            "prev_net": prev_net, "prev_raw": prev_raw})
            bear_done = True
        if not bull_done and c[i-1] < orb_low and c[i] > orb_low and c[i] > o[i]:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                            "signal": "BullishReject", "entry_price": c[i],
                            "prev_net": prev_net, "prev_raw": prev_raw})
            bull_done = True
        if not bear_done and i >= 2 and h[i] < h[i-1] < h[i-2] and c[i] < orb_high:
            records.append({"day": day, "entry_time": idx[i], "direction": "SHORT",
                            "signal": "LowerHigh", "entry_price": c[i],
                            "prev_net": prev_net, "prev_raw": prev_raw})
            bear_done = True
        if not bull_done and i >= 2 and lo[i] > lo[i-1] > lo[i-2] and c[i] > orb_low:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                            "signal": "HigherLow", "entry_price": c[i],
                            "prev_net": prev_net, "prev_raw": prev_raw})
            bull_done = True

entries_df = pd.DataFrame(records).dropna(subset=["prev_net"]).reset_index(drop=True)

# Classify: continuation vs fade (only meaningful for big-trend bucket)
# SHORT + yesterday bearish (prev_raw < 0) = continuation
# LONG  + yesterday bullish (prev_raw > 0) = continuation
entries_df["alignment"] = np.where(
    ((entries_df["direction"] == "SHORT") & (entries_df["prev_raw"] < 0)) |
    ((entries_df["direction"] == "LONG")  & (entries_df["prev_raw"] > 0)),
    "continuation", "fade"
)

print(f"Total signals: {len(entries_df)}")


# ── Simulate exits ────────────────────────────────────────────────────────────
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
    print(f"\n── {label} ─────────────────────────────────────────────────────────")
    print(f"  Trades        : {n}")
    print(f"  Win rate      : {wr:.1f}%")
    print(f"  Avg P&L       : {avg:+.2f} pts")
    print(f"  Total P&L     : {df_sub['pnl_pts'].sum():+.1f} pts")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  Exits         : {exits}")
    return {"n": n, "wr": wr, "avg": avg, "total": df_sub["pnl_pts"].sum(), "pf": pf}


# ── Run analysis ──────────────────────────────────────────────────────────────
all_trades = simulate(entries_df, df)
all_trades.to_csv(os.path.join(OUT_DIR, "all_trades.csv"), index=False)

quiet   = entries_df[entries_df["prev_net"] <= PREV_MOVE_THRESHOLD].reset_index(drop=True)
bigtren = entries_df[entries_df["prev_net"] >  PREV_MOVE_THRESHOLD].reset_index(drop=True)

cont_entries = bigtren[bigtren["alignment"] == "continuation"].reset_index(drop=True)
fade_entries = bigtren[bigtren["alignment"] == "fade"].reset_index(drop=True)

print("\n" + "=" * 70)
print("REFERENCE BUCKETS")
print("=" * 70)
summarise(all_trades, "Unfiltered (all signals)")
summarise(simulate(quiet, df),   f"Quiet prev-day (<= {PREV_MOVE_THRESHOLD}%) — currently ALLOWED")
summarise(simulate(bigtren, df), f"Big-trend prev-day (> {PREV_MOVE_THRESHOLD}%) — currently BLOCKED")

print("\n" + "=" * 70)
print(f"BIG-TREND BUCKET SPLIT BY DIRECTION ALIGNMENT")
print(f"(only trades where prev_net > {PREV_MOVE_THRESHOLD}%)")
print("=" * 70)
cont_trades = simulate(cont_entries, df)
fade_trades = simulate(fade_entries, df)
cont_trades.to_csv(os.path.join(OUT_DIR, "continuation_trades.csv"), index=False)
fade_trades.to_csv(os.path.join(OUT_DIR, "fade_trades.csv"), index=False)

summarise(cont_trades, "Continuation (signal aligns with yesterday's direction)")
summarise(fade_trades, "Fade (signal opposes yesterday's direction)")

# Further split continuation by signal direction
print("\n" + "=" * 70)
print("CONTINUATION — BEAR vs BULL breakdown")
print("=" * 70)
summarise(simulate(cont_entries[cont_entries["direction"] == "SHORT"].reset_index(drop=True), df),
          "Continuation BEAR (yesterday fell, today SHORT)")
summarise(simulate(cont_entries[cont_entries["direction"] == "LONG"].reset_index(drop=True), df),
          "Continuation BULL (yesterday rose, today LONG)")

print("\n" + "=" * 70)
print("FADE — BEAR vs BULL breakdown")
print("=" * 70)
summarise(simulate(fade_entries[fade_entries["direction"] == "SHORT"].reset_index(drop=True), df),
          "Fade BEAR (yesterday rose, today SHORT)")
summarise(simulate(fade_entries[fade_entries["direction"] == "LONG"].reset_index(drop=True), df),
          "Fade BULL (yesterday fell, today LONG)")

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
rows = [
    ("Unfiltered", all_trades),
    (f"Quiet (<=0.42%)", simulate(quiet, df)),
    (f"Big-trend (>0.42%)", simulate(bigtren, df)),
    ("  Continuation", cont_trades),
    ("  Fade", fade_trades),
]
print(f"  {'Bucket':<30} {'N':>6} {'WR%':>7} {'Avg':>8} {'PF':>7}")
print("  " + "-" * 62)
for label, trades in rows:
    if trades.empty:
        print(f"  {label:<30} {'—':>6}")
        continue
    n  = len(trades)
    wr = (trades["pnl_pts"] > 0).mean() * 100
    avg = trades["pnl_pts"].mean()
    gw = trades[trades["pnl_pts"] > 0]["pnl_pts"].sum()
    gl = abs(trades[trades["pnl_pts"] <= 0]["pnl_pts"].sum())
    pf = gw / gl if gl > 0 else float("inf")
    print(f"  {label:<30} {n:>6} {wr:>7.1f} {avg:>8.2f} {pf:>7.2f}")

print("\nCSVs saved to:", OUT_DIR)
print("Backtest complete.")
