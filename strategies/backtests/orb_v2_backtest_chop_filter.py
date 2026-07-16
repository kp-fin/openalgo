"""
ORB_Spread — Choppiness Index Regime Filter Test
Nifty 50 Spot | 2021-07-01 → 2026-06-30

Follow-up to the ADX >= 18 test (rejected for this strategy — see
orb_spread.md "Signal Filter Testing"). ADX gates for TREND STRENGTH,
which is backwards for a strategy that fades FAILED breakouts: a fade
strategy should plausibly want evidence of a CHOPPY/range-bound regime,
not a trending one. Choppiness Index (CHOP) is the direct counterpart —
same category of indicator, opposite logical direction.

CHOP formula (Dreiss), 0-100 scale, period n:
    CHOP = 100 * log10( sum(TR, n) / (HH_n - LL_n) ) / log10(n)
High CHOP (near 100) = choppy/sideways. Low CHOP (near 0) = strongly
trending (net range ~= sum of true ranges, i.e. consistent one-way
movement with little backtrack).

Hypothesis: CHOP >= threshold (favoring choppy days) should suit
ORB_Spread's rejection/fade signals better than the ADX trend filter did.

Computed continuously (simple rolling sum/max/min, no smoothing beyond
what's built into the formula) on 5m NIFTY bars, period 14 (matches the
ADX test's period for consistency), evaluated at the last OR-window bar
(09:40, just before the 09:45 entry window opens — causal, no lookahead).

Baseline is the CURRENT live config (HARD_EXIT 15:15, adopted 2026-07-16),
not the older 14:30 config, so this comparison reflects what's actually
deployed. Entry logic and TARGET_PTS/STOP_PTS unchanged throughout.
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

CHOP_PERIOD     = 14
CHOP_THRESHOLDS = [50, 61.8]   # midpoint and the common Fibonacci-based default

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orb_v2_chop_filter")
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

# ── Choppiness Index (continuous, no per-day reset, causal) ──────────────────
prev_close = df["close"].shift(1)
tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs()], axis=1).max(axis=1)

tr_sum = tr.rolling(CHOP_PERIOD).sum()
hh     = df["high"].rolling(CHOP_PERIOD).max()
ll     = df["low"].rolling(CHOP_PERIOD).min()
rng    = (hh - ll).replace(0, np.nan)  # avoid div-by-zero on dead-flat windows

df["chop"] = 100 * np.log10(tr_sum / rng) / np.log10(CHOP_PERIOD)

print(f"CHOP stats (all bars, post-warmup): mean={df['chop'].mean():.1f}, "
      f"median={df['chop'].median():.1f}, std={df['chop'].std():.1f}")

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

    chop_at_or = or_window["chop"].iloc[-1] if not or_window.empty else np.nan

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
                             "signal": "BearishReject", "entry_price": c[i], "chop": chop_at_or})
            bear_done = True
        if not bull_done and c[i-1] < orb_low and c[i] > orb_low and c[i] > o[i]:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                             "signal": "BullishReject", "entry_price": c[i], "chop": chop_at_or})
            bull_done = True
        if not bear_done and i >= 2 and h[i] < h[i-1] < h[i-2] and c[i] < orb_high:
            records.append({"day": day, "entry_time": idx[i], "direction": "SHORT",
                             "signal": "LowerHigh", "entry_price": c[i], "chop": chop_at_or})
            bear_done = True
        if not bull_done and i >= 2 and lo[i] > lo[i-1] > lo[i-2] and c[i] > orb_low:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                             "signal": "HigherLow", "entry_price": c[i], "chop": chop_at_or})
            bull_done = True

entries_df = pd.DataFrame(records)
n_before = len(entries_df)
entries_df = entries_df.dropna(subset=["chop"]).reset_index(drop=True)
print(f"Signals generated: {n_before} | dropped {n_before - len(entries_df)} with no CHOP warmup yet")
print(f"CHOP at entry — mean: {entries_df['chop'].mean():.1f}, median: {entries_df['chop'].median():.1f}")


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
print("BASELINE — no CHOP filter")
print("=" * 90)
unfiltered = simulate(entries_df, df)
res_all = summarise(unfiltered, "Unfiltered")
unfiltered.to_csv(os.path.join(OUT_DIR, "unfiltered.csv"), index=False)

results = {"Unfiltered": res_all}
for thresh in CHOP_THRESHOLDS:
    filtered_entries = entries_df[entries_df["chop"] >= thresh].reset_index(drop=True)
    filtered = simulate(filtered_entries, df)
    label = f"CHOP>={thresh}"
    print("\n" + "=" * 90)
    print(f"{label} ({len(filtered_entries)}/{len(entries_df)} signals pass)")
    print("=" * 90)
    results[label] = summarise(filtered, label)
    filtered.to_csv(os.path.join(OUT_DIR, f"chop_ge_{str(thresh).replace('.', '')}.csv"), index=False)

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
