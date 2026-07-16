"""
ORB_Spread — Opening Gap Size Regime Filter Test
Nifty 50 Spot | 2021-07-01 → 2026-06-30

Third regime filter, distinct mechanism from CHOP/BB-Width (both
rejected/mixed — see orb_spread.md "Signal Filter Testing"). Those
looked at price action AFTER the session opens; this looks BEFORE the
OR even forms: the overnight gap (today's open vs. yesterday's close).

Hypothesis: a large gap often predicts a trend/continuation day (bad
for a strategy that fades failed breakouts -- if today is genuinely
trending, breakouts are less likely to fail); a small gap often
predicts a range day (good for fades). Filtering OUT large-gap days
should help.

Gap size = |today's 09:15 open - yesterday's last close| / yesterday's
close, expressed as a percentage. Known and available before the OR
even starts forming, so this is the most causal of the filters tried
so far -- no rolling-window warmup or same-day computation needed.

Baseline is the CURRENT live config (HARD_EXIT 15:15, adopted
2026-07-16). Entry logic and TARGET_PTS/STOP_PTS unchanged throughout.
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

GAP_PCT_MAX_LEVELS = [0.25, 0.40]   # keep only days with gap <= this % (filter OUT big gaps)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orb_v2_gap_filter")
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
print(f"Loaded {len(df):,} 5m bars: {df.index[0]} → {df.index[-1]}")

# ── Gap size per day: today's 09:15 open vs. yesterday's last close ──────────
daily_gaps = []
prev_close = None
for day, grp in df.groupby(df.index.date):
    first_bar = grp.between_time("09:15", "09:15")
    if first_bar.empty:
        continue
    today_open = first_bar["open"].iloc[0]
    if prev_close is not None:
        gap_pct = abs(today_open - prev_close) / prev_close * 100
        daily_gaps.append({"day": day, "gap_pct": gap_pct})
    prev_close = grp["close"].iloc[-1]

daily_gap_df = pd.DataFrame(daily_gaps).set_index("day")
print(f"Gap%% stats: mean={daily_gap_df['gap_pct'].mean():.3f}%, "
      f"median={daily_gap_df['gap_pct'].median():.3f}%, "
      f"90th pctile={daily_gap_df['gap_pct'].quantile(0.9):.3f}%")

# ── Signal generation (identical to orb_v2_backtest.py) ──────────────────────
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

    gap_today = daily_gap_df["gap_pct"].get(day, np.nan)

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
                             "signal": "BearishReject", "entry_price": c[i], "gap_pct": gap_today})
            bear_done = True
        if not bull_done and c[i-1] < orb_low and c[i] > orb_low and c[i] > o[i]:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                             "signal": "BullishReject", "entry_price": c[i], "gap_pct": gap_today})
            bull_done = True
        if not bear_done and i >= 2 and h[i] < h[i-1] < h[i-2] and c[i] < orb_high:
            records.append({"day": day, "entry_time": idx[i], "direction": "SHORT",
                             "signal": "LowerHigh", "entry_price": c[i], "gap_pct": gap_today})
            bear_done = True
        if not bull_done and i >= 2 and lo[i] > lo[i-1] > lo[i-2] and c[i] > orb_low:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                             "signal": "HigherLow", "entry_price": c[i], "gap_pct": gap_today})
            bull_done = True

entries_df = pd.DataFrame(records)
n_before = len(entries_df)
entries_df = entries_df.dropna(subset=["gap_pct"]).reset_index(drop=True)
print(f"Signals generated: {n_before} | dropped {n_before - len(entries_df)} with no prior-day close available")


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
    print(f"  Exits         : {df_sub['reason'].value_counts().to_dict()}")
    return {"n": n, "wr": wr, "avg": avg, "total": df_sub["pnl_pts"].sum(), "pf": pf}


print("\n" + "=" * 90)
print("BASELINE — no gap filter")
print("=" * 90)
unfiltered = simulate(entries_df, df)
res_all = summarise(unfiltered, "Unfiltered")
unfiltered.to_csv(os.path.join(OUT_DIR, "unfiltered.csv"), index=False)

results = {"Unfiltered": res_all}
for max_gap in GAP_PCT_MAX_LEVELS:
    filtered_entries = entries_df[entries_df["gap_pct"] <= max_gap].reset_index(drop=True)
    filtered = simulate(filtered_entries, df)
    label = f"Gap<={max_gap}%"
    print("\n" + "=" * 90)
    print(f"{label} ({len(filtered_entries)}/{len(entries_df)} signals pass)")
    print("=" * 90)
    results[label] = summarise(filtered, label)
    filtered.to_csv(os.path.join(OUT_DIR, f"gap_le_{str(max_gap).replace('.', '')}.csv"), index=False)

print("\n" + "=" * 90)
print("COMPARISON")
print("=" * 90)
labels = list(results.keys())
print(f"  {'Metric':<15}" + "".join(f"{l:>15}" for l in labels))
for metric, fmt in [("n", "{:>15d}"), ("wr", "{:>15.1f}"), ("avg", "{:>15.2f}"),
                     ("total", "{:>15.1f}"), ("pf", "{:>15.2f}")]:
    row = f"  {metric:<15}"
    for l in labels:
        v = results.get(l, {}).get(metric)
        row += fmt.format(v) if v is not None else f"{'—':>15}"
    print(row)

print("\nBacktest complete.")
