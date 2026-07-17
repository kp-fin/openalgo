"""
ORB_Spread — Day-of-Week / Expiry-Day Re-Test, Regime-Aware
Nifty 50 Spot | 2021-07-01 -> 2026-06-30

Re-run of the 2026-07-16 Day-of-Week test (orb_v2_backtest_prevday_dow_filter.py,
PART 2), corrected for NSE's actual expiry-day switch. The original test
labeled "Thursday" as the expiry day across the WHOLE 5-year window -- wrong,
since NIFTY's weekly/monthly expiry moved from Thursday to **Tuesday on
2025-09-01** (Karan confirmed the exact date, 2026-07-17). A blanket label
across a regime change is invalid: most of this backtest's history (2021-07-01
-> 2025-08-31, ~4 years) really was a Thursday-expiry regime, but the last
~10 months (2025-09-01 -> 2026-06-30) are Tuesday-expiry.

This test classifies each trade's expiry-day status using the REGIME ACTUALLY
IN EFFECT on that trade's date:
    date <  2025-09-01 -> expiry day = Thursday (weekday() == 3)
    date >= 2025-09-01 -> expiry day = Tuesday  (weekday() == 1)

Reports three views:
  1. Regime-aware expiry-day vs non-expiry-day split (the corrected version
     of the original hypothesis) across the full 5-year window.
  2. The same split broken out separately within each regime period, since
     the new Tuesday-regime sample is necessarily much thinner (~10 months
     vs ~4 years) -- flagged explicitly, not glossed over.
  3. The original naive 5-way calendar-weekday split, kept for direct
     comparison to the 2026-07-16 write-up (same underlying trade set).

Entry signal logic, OR/range-day filters, and exit rules (TARGET/STOP/HARD_EXIT)
are identical to every other ORB_Spread backtest this vault has run -- current
live config: HARD_EXIT 15:15, TARGET_PTS 40, STOP_PTS 25. Only the expiry-day
classification changes vs. the 2026-07-16 test.
"""

import os
import warnings
from datetime import date as ddate, time as dtime

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

EXPIRY_SWITCH_DATE = ddate(2025, 9, 1)   # NIFTY weekly/monthly expiry: Thu -> Tue
OLD_EXPIRY_WEEKDAY = 3   # Thursday (Mon=0 ... Sun=6)
NEW_EXPIRY_WEEKDAY = 1   # Tuesday

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orb_v2_dow_expiry_regime_aware")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Fetch data ────────────────────────────────────────────────────────────────
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


def regime_and_is_expiry(day: ddate) -> tuple:
    if day < EXPIRY_SWITCH_DATE:
        return "old (Thursday)", day.weekday() == OLD_EXPIRY_WEEKDAY
    return "new (Tuesday)", day.weekday() == NEW_EXPIRY_WEEKDAY


# ── Signal generation (identical to every other ORB_Spread backtest) ─────────
records = []
for day, grp in df.groupby(df.index.date):
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

    regime, is_expiry = regime_and_is_expiry(day)
    dow = pd.Timestamp(day).day_name()

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
                             "dow": dow, "regime": regime, "is_expiry": is_expiry})
            bear_done = True
        if not bull_done and c[i-1] < orb_low and c[i] > orb_low and c[i] > o[i]:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                             "signal": "BullishReject", "entry_price": c[i],
                             "dow": dow, "regime": regime, "is_expiry": is_expiry})
            bull_done = True
        if not bear_done and i >= 2 and h[i] < h[i-1] < h[i-2] and c[i] < orb_high:
            records.append({"day": day, "entry_time": idx[i], "direction": "SHORT",
                             "signal": "LowerHigh", "entry_price": c[i],
                             "dow": dow, "regime": regime, "is_expiry": is_expiry})
            bear_done = True
        if not bull_done and i >= 2 and lo[i] > lo[i-1] > lo[i-2] and c[i] > orb_low:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                             "signal": "HigherLow", "entry_price": c[i],
                             "dow": dow, "regime": regime, "is_expiry": is_expiry})
            bull_done = True

entries_df = pd.DataFrame(records)
print(f"Signals generated: {len(entries_df)}")
print(f"Regime split: {entries_df['regime'].value_counts().to_dict()}")
print(f"Expiry-day trades (regime-aware): {entries_df['is_expiry'].sum()} of {len(entries_df)}")


def simulate(entries_df, df):
    out = entries_df.copy()
    out["exit_time"], out["exit_price"], out["pnl_pts"], out["reason"] = pd.NaT, np.nan, np.nan, ""

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

            out.at[row_i, "exit_time"], out.at[row_i, "exit_price"] = ts, px
            out.at[row_i, "pnl_pts"], out.at[row_i, "reason"] = pnl, reason
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
    print(f"\n── {label} ─────────────────────────────────────")
    print(f"  Trades        : {n}")
    print(f"  Win rate      : {wr:.1f}%")
    print(f"  Avg P&L       : {avg:+.2f} pts")
    print(f"  Total P&L     : {df_sub['pnl_pts'].sum():+.1f} pts")
    print(f"  Profit factor : {pf:.2f}")
    return {"n": n, "wr": wr, "avg": avg, "total": df_sub["pnl_pts"].sum(), "pf": pf}


all_trades = simulate(entries_df, df)
all_trades.to_csv(os.path.join(OUT_DIR, "all_trades.csv"), index=False)

print("\n" + "=" * 90)
print("PART 1: REGIME-AWARE EXPIRY-DAY SPLIT (full 5-year window)")
print("=" * 90)
res_all     = summarise(all_trades, "Unfiltered (all 5 years)")
expiry_df   = entries_df[entries_df["is_expiry"]].reset_index(drop=True)
nonexpiry_df = entries_df[~entries_df["is_expiry"]].reset_index(drop=True)
res_expiry    = summarise(simulate(expiry_df, df), "Expiry day (regime-aware: Thu pre-2025-09-01, Tue after)")
res_nonexpiry = summarise(simulate(nonexpiry_df, df), "Non-expiry day")

print("\n" + "=" * 90)
print("PART 2: SAME SPLIT, WITHIN EACH REGIME PERIOD SEPARATELY")
print("=" * 90)
for regime_label in ["old (Thursday)", "new (Tuesday)"]:
    regime_entries = entries_df[entries_df["regime"] == regime_label]
    n_days = regime_entries["day"].nunique()
    print(f"\n--- Regime: {regime_label} ({n_days} distinct trading days with a signal in this regime) ---")
    r_expiry    = regime_entries[regime_entries["is_expiry"]].reset_index(drop=True)
    r_nonexpiry = regime_entries[~regime_entries["is_expiry"]].reset_index(drop=True)
    summarise(simulate(r_expiry, df), f"{regime_label} — expiry day")
    summarise(simulate(r_nonexpiry, df), f"{regime_label} — non-expiry day")

print("\n" + "=" * 90)
print("PART 3: ORIGINAL NAIVE 5-WAY CALENDAR-WEEKDAY SPLIT (for direct comparison to 2026-07-16 write-up)")
print("=" * 90)
dow_results = {}
for dow in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
    dow_entries = entries_df[entries_df["dow"] == dow].reset_index(drop=True)
    if dow_entries.empty:
        continue
    dow_results[dow] = summarise(simulate(dow_entries, df), dow)

print("\n" + "=" * 90)
print("COMPARISON — Regime-aware expiry vs non-expiry (full window)")
print("=" * 90)
labels1 = ["Unfiltered", "Expiry day", "Non-expiry day"]
res1 = [res_all, res_expiry, res_nonexpiry]
print(f"  {'Metric':<15}" + "".join(f"{l:>18}" for l in labels1))
for metric, fmt in [("n", "{:>18d}"), ("wr", "{:>18.1f}"), ("avg", "{:>18.2f}"), ("pf", "{:>18.2f}")]:
    row = f"  {metric:<15}"
    for r in res1:
        row += fmt.format(r[metric]) if r else f"{'—':>18}"
    print(row)

print("\n" + "=" * 90)
print("COMPARISON — Original naive weekday split")
print("=" * 90)
print(f"  {'Day':<12}{'Trades':>8}{'WR%':>8}{'Avg':>8}{'PF':>8}")
for dow, res in dow_results.items():
    if res:
        print(f"  {dow:<12}{res['n']:>8}{res['wr']:>8.1f}{res['avg']:>8.2f}{res['pf']:>8.2f}")

print("\nBacktest complete.")
