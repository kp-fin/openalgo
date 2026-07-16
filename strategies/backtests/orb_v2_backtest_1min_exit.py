"""
ORB_Spread — 1-Minute Exit-Check Resolution Test
Nifty 50 Spot | 2021-07-01 → 2026-06-30

Motivated by a live observation: today's (2026-07-16) ORB_Spread bear
put spread was at -24.5pts at the 10:50 check (a hair from the -25pt
stop) but wasn't re-checked until 10:55, by which point spot had
slipped to -30.3pts before the stop actually fired -- pure slippage
from the 5-minute gap between exit checks, not from the strategy logic.

This tests whether checking exits against 1-minute closes instead of
5-minute closes would have changed the historical backtest numbers.

IMPORTANT — what does NOT change: entry signals are generated from the
exact same 5-minute-bar entry logic as every other ORB_Spread backtest
(identical 794-signal set) -- only how often the OPEN POSITION is
re-checked for target/stop/hard-exit changes. This isolates exit-check
granularity as the only variable, so it's a fair comparison against the
already-established 5-min baseline (PF 1.16, avg P&L +2.99, from the
hard-exit-time test).

Baseline is the CURRENT live config (HARD_EXIT 15:15, TARGET_PTS 40,
STOP_PTS 25). Two 5-year datasets fetched: native 5m bars (for entry
signals, unchanged) and native 1m bars (for the exit walk, new).
"""

import os
import warnings
from datetime import time as dtime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY    = os.getenv("OPENALGO_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENALGO_API_KEY environment variable before running.")
HOST       = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
START_DATE = "2021-07-01"
END_DATE   = "2026-06-30"

OR_MIN, OR_MAX = 30, 150
ENTRY_END = dtime(12, 0)
HARD_EXIT = dtime(15, 15)   # current live config, adopted 2026-07-16
RANGE_CHK = dtime(10, 15)
TARGET_PTS, STOP_PTS = 40, 25

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orb_v2_1min_exit")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Fetch data ────────────────────────────────────────────────────────────────
from openalgo import api as openalgo_api
import pytz

IST    = pytz.timezone("Asia/Kolkata")
client = openalgo_api(api_key=API_KEY, host=HOST)


def _fetch(interval):
    resp = client.history(symbol="NIFTY", exchange="NSE_INDEX", interval=interval,
                           start_date=START_DATE, end_date=END_DATE)
    if isinstance(resp, dict):
        if resp.get("status") != "success":
            raise SystemExit(f"API error ({interval}): {resp}")
        out = pd.DataFrame(resp.get("data", []))
        out["datetime"] = pd.to_datetime(out["datetime"])
        out = out.set_index("datetime")
    elif hasattr(resp, "empty"):
        out = resp
    else:
        raise SystemExit(f"Unexpected response type ({interval}): {type(resp)}")
    out.columns = [c.lower() for c in out.columns]
    out.index = out.index.tz_localize("Asia/Kolkata") if out.index.tz is None else out.index.tz_convert(IST)
    return out.sort_index()


print(f"Fetching 5m Nifty data (entry signals, unchanged) ...")
df5 = _fetch("5m")
print(f"Loaded {len(df5):,} 5m bars: {df5.index[0]} → {df5.index[-1]}")

print(f"Fetching 1m Nifty data (exit-check resolution) ...")
df1 = _fetch("1m")
print(f"Loaded {len(df1):,} 1m bars: {df1.index[0]} → {df1.index[-1]}")

# ── Signal generation on 5m bars (identical to every other ORB_Spread test) ──
records = []
for day, grp in df5.groupby(df5.index.date):
    or_window = grp.between_time("09:15", "09:44")
    if len(or_window) < 3:
        continue
    orb_high, orb_low = or_window["high"].max(), or_window["low"].min()
    or_range = orb_high - orb_low
    if or_range < OR_MIN or or_range > OR_MAX:
        continue

    at_1015 = grp.between_time("10:15", "10:15")
    if not at_1015.empty:
        c1015 = at_1015["close"].iloc[0]
        if orb_low < c1015 < orb_high:
            continue

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
                             "signal": "BearishReject", "entry_price": c[i]})
            bear_done = True
        if not bull_done and c[i-1] < orb_low and c[i] > orb_low and c[i] > o[i]:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                             "signal": "BullishReject", "entry_price": c[i]})
            bull_done = True
        if not bear_done and i >= 2 and h[i] < h[i-1] < h[i-2] and c[i] < orb_high:
            records.append({"day": day, "entry_time": idx[i], "direction": "SHORT",
                             "signal": "LowerHigh", "entry_price": c[i]})
            bear_done = True
        if not bull_done and i >= 2 and lo[i] > lo[i-1] > lo[i-2] and c[i] > orb_low:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                             "signal": "HigherLow", "entry_price": c[i]})
            bull_done = True

entries_df = pd.DataFrame(records)
print(f"Signals generated: {len(entries_df)} (identical entry set for both resolutions)")


def simulate(entries_df, df, label):
    """df: the bar series to walk for exit-checking (either df5 or df1).
    Note entry_price/entry_time still come from the 5m signal generation
    above -- only the exit walk resolution changes."""
    out = entries_df.copy()
    out["exit_time"], out["exit_price"], out["pnl_pts"], out["reason"] = pd.NaT, np.nan, np.nan, ""
    out["overshoot"] = np.nan   # how far past the exact threshold the exit landed

    for row_i, row in out.iterrows():
        entry_px = row["entry_price"]
        sign     = 1 if row["direction"] == "SHORT" else -1
        day_bars = df[df.index.date == row["day"]]
        walk = day_bars[day_bars.index >= row["entry_time"]]

        for ts, bar in walk.iterrows():
            if ts == row["entry_time"]:
                continue
            px  = bar["close"]
            pnl = (entry_px - px) * sign
            t   = ts.time()

            if pnl >= TARGET_PTS:
                reason, overshoot = "TARGET", pnl - TARGET_PTS
            elif pnl <= -STOP_PTS:
                reason, overshoot = "STOP", (-pnl) - STOP_PTS
            elif t >= HARD_EXIT:
                reason, overshoot = "HARD_EXIT", 0.0
            else:
                continue

            out.at[row_i, "exit_time"], out.at[row_i, "exit_price"] = ts, px
            out.at[row_i, "pnl_pts"], out.at[row_i, "reason"] = pnl, reason
            out.at[row_i, "overshoot"] = overshoot
            break

    return out.dropna(subset=["pnl_pts"]).copy()


def summarise(df_sub, label):
    n    = len(df_sub)
    wins = (df_sub["pnl_pts"] > 0).sum()
    wr   = wins / n * 100
    avg  = df_sub["pnl_pts"].mean()
    gw   = df_sub[df_sub["pnl_pts"] > 0]["pnl_pts"].sum()
    gl   = abs(df_sub[df_sub["pnl_pts"] <= 0]["pnl_pts"].sum())
    pf   = gw / gl if gl > 0 else float("inf")
    exits = df_sub["reason"].value_counts().to_dict()

    stop_rows = df_sub[df_sub["reason"] == "STOP"]
    target_rows = df_sub[df_sub["reason"] == "TARGET"]
    avg_stop_overshoot = stop_rows["overshoot"].mean() if len(stop_rows) else 0
    avg_target_overshoot = target_rows["overshoot"].mean() if len(target_rows) else 0

    print(f"\n── {label} ─────────────────────────────────────")
    print(f"  Trades        : {n}")
    print(f"  Win rate      : {wr:.1f}%")
    print(f"  Avg P&L       : {avg:+.2f} pts")
    print(f"  Total P&L     : {df_sub['pnl_pts'].sum():+.1f} pts")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  Exits         : {exits}")
    print(f"  Avg STOP overshoot   (pts past -25)  : {avg_stop_overshoot:.2f}")
    print(f"  Avg TARGET overshoot (pts past +40)  : {avg_target_overshoot:.2f}")
    return {"n": n, "wr": wr, "avg": avg, "total": df_sub["pnl_pts"].sum(), "pf": pf,
            "stop_overshoot": avg_stop_overshoot, "target_overshoot": avg_target_overshoot}


print("\n" + "=" * 90)
print("5-MINUTE exit checking (current live behavior)")
print("=" * 90)
trades_5m = simulate(entries_df, df5, "5m exit-check")
res_5m = summarise(trades_5m, "5m exit-check")
trades_5m.to_csv(os.path.join(OUT_DIR, "exit_5min.csv"), index=False)

print("\n" + "=" * 90)
print("1-MINUTE exit checking (proposed)")
print("=" * 90)
trades_1m = simulate(entries_df, df1, "1m exit-check")
res_1m = summarise(trades_1m, "1m exit-check")
trades_1m.to_csv(os.path.join(OUT_DIR, "exit_1min.csv"), index=False)

print("\n" + "=" * 90)
print("COMPARISON")
print("=" * 90)
print(f"  {'Metric':<25}{'5-min exit':>15}{'1-min exit':>15}")
for metric, fmt, label in [
    ("n", "{:>15d}", "Trades"),
    ("wr", "{:>15.1f}", "Win rate %"),
    ("avg", "{:>15.2f}", "Avg P&L pts"),
    ("total", "{:>15.1f}", "Total P&L pts"),
    ("pf", "{:>15.2f}", "Profit factor"),
    ("stop_overshoot", "{:>15.2f}", "Avg STOP overshoot"),
    ("target_overshoot", "{:>15.2f}", "Avg TARGET overshoot"),
]:
    print(f"  {label:<25}{fmt.format(res_5m[metric])}{fmt.format(res_1m[metric])}")

print("\nBacktest complete.")
