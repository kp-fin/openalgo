"""
ORB v2 — Intraday-ATR Trailing Stop Variant — VectorBT-style Backtest
Nifty 50 Spot | 5m candles | 2021-07-01 → 2026-06-30

Follow-up to orb_v2_backtest_atr_trail.py. That first pass used DAILY ATR
for the trailing distance and found the stop essentially never engaged
(779/785 trades rode to the 14:30 hard exit, only 6 hit the trailing
stop) — daily ATR (~150-300+ pts) is far wider than a realistic same-day
intraday excursion from a post-opening-range entry, so the "improvement"
in that run was really just "remove the tight stop," not evidence that
ATR-trailing itself works.

This variant fixes the scale mismatch: ATR is computed on the 5m bar
series itself (continuous across the whole history, not reset per day,
so it has real warmup and isn't just the 6 bars available before a day's
signal window opens), which puts the stop distance on the same order of
magnitude as intraday moves.

  Overnight gap handling: the first 5m bar of each trading day uses
  TR = high - low only (no |high - prev_close| / |low - prev_close|
  gap component) so the ATR isn't inflated by overnight gap noise —
  that's a daily-timeframe phenomenon, not an intraday one.

  ATR is smoothed continuously (Wilder RMA) across the full 5m series,
  so by the time any real trading day's signals fire, the ATR estimate
  reflects many prior days of 5m bar-to-bar movement, not just today's
  thin opening-range sample.

Entry logic and everything else (ATR_MULT, trailing mechanic, hard exit,
no fixed target, side-by-side comparison against the fixed baseline) is
identical to orb_v2_backtest_atr_trail.py — only the ATR timeframe changes.

P&L in spot points (proxy), same convention as the baseline backtest.
"""

import os
import warnings
from datetime import time as dtime

import numpy as np
import pandas as pd
import quantstats as qs

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY    = os.getenv("OPENALGO_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENALGO_API_KEY environment variable before running.")
HOST       = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
START_DATE = "2021-07-01"
END_DATE   = "2026-06-30"

OR_MIN     = 30    # skip day if 30m OR range below this (pts)
OR_MAX     = 150   # skip day if 30m OR range above this (pts)
ENTRY_END  = dtime(12, 0)
HARD_EXIT  = dtime(14, 30)
RANGE_CHK  = dtime(10, 15)

# Baseline (fixed) exit params — replicated here for side-by-side comparison
TARGET_PTS = 40
STOP_PTS   = 25

# ATR-trailing variant params — same multiplier as the daily-ATR pass,
# only the ATR timeframe (5m bars vs. daily bars) changes.
ATR_PERIOD = 14    # 5m bars (~70 min of smoothing input per step)
ATR_MULT   = 1.5   # trailing/initial stop distance = ATR_MULT * intraday ATR

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orb_v2_atr_trail_intraday")
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
            raise RuntimeError(f"API error ({interval}): {resp}")
        records = resp.get("data", [])
        out = pd.DataFrame(records)
        out["datetime"] = pd.to_datetime(out["datetime"])
        out = out.set_index("datetime")
    elif hasattr(resp, "empty"):
        out = resp
    else:
        raise RuntimeError(f"Unexpected response type ({interval}): {type(resp)}")

    if out is None or out.empty:
        raise RuntimeError(f"Empty response from OpenAlgo for interval={interval}.")

    out.columns = [c.lower() for c in out.columns]
    if out.index.tz is None:
        out.index = out.index.tz_localize("Asia/Kolkata")
    else:
        out.index = out.index.tz_convert(IST)
    return out.sort_index()


try:
    print(f"Fetching 5m Nifty data {START_DATE} → {END_DATE} ...")
    df = _fetch("5m")
    print(f"Loaded {len(df):,} intraday bars: {df.index[0]} → {df.index[-1]}")
except Exception as exc:
    raise SystemExit(f"Data fetch failed: {exc}")

# ── Intraday ATR (Wilder RMA on 5m bars), overnight gap excluded ─────────────
day_of_bar   = df.index.date
is_first_bar = pd.Series(day_of_bar, index=df.index).ne(pd.Series(day_of_bar, index=df.index).shift(1))

prev_close = df["close"].shift(1)
gap_hi = (df["high"] - prev_close).abs()
gap_lo = (df["low"]  - prev_close).abs()
# For the first bar of each day, drop the gap components (no overnight TR).
gap_hi = gap_hi.where(~is_first_bar, 0.0)
gap_lo = gap_lo.where(~is_first_bar, 0.0)

tr = pd.concat([df["high"] - df["low"], gap_hi, gap_lo], axis=1).max(axis=1)


def wilder_rma(series, period):
    rma = pd.Series(np.nan, index=series.index)
    if len(series) <= period:
        return rma
    rma.iloc[period] = series.iloc[1:period + 1].mean()
    vals = series.values
    rma_vals = rma.values
    for i in range(period + 1, len(series)):
        rma_vals[i] = (rma_vals[i - 1] * (period - 1) + vals[i]) / period
    return pd.Series(rma_vals, index=series.index)


df["atr"] = wilder_rma(tr, ATR_PERIOD)

# ── Signal generation (identical to baseline orb_v2_backtest.py) ─────────────
records = []
total_days, skip_range, skip_range_day = 0, 0, 0

for day, grp in df.groupby(df.index.date):
    total_days += 1

    or_window = grp.between_time("09:15", "09:44")
    if len(or_window) < 3:
        continue
    orb_high = or_window["high"].max()
    orb_low  = or_window["low"].min()
    or_range = orb_high - orb_low

    if or_range < OR_MIN or or_range > OR_MAX:
        skip_range += 1
        continue

    at_1015 = grp.between_time("10:15", "10:15")
    if not at_1015.empty:
        c1015 = at_1015["close"].iloc[0]
        if orb_low < c1015 < orb_high:
            skip_range_day += 1
            continue

    sig = grp.between_time("09:45", "12:00")
    if len(sig) < 2:
        continue

    c, o, h, lo = sig["close"].values, sig["open"].values, sig["high"].values, sig["low"].values
    idx = sig.index
    bear_done = bull_done = False

    for i in range(1, len(sig)):
        t = idx[i].time()
        if t > ENTRY_END:
            break

        if not bear_done and c[i-1] > orb_high and c[i] < orb_high and c[i] < o[i]:
            records.append({"day": day, "entry_time": idx[i], "direction": "SHORT",
                             "signal": "BearishReject", "entry_price": c[i],
                             "orb_high": orb_high, "orb_low": orb_low})
            bear_done = True

        if not bull_done and c[i-1] < orb_low and c[i] > orb_low and c[i] > o[i]:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                             "signal": "BullishReject", "entry_price": c[i],
                             "orb_high": orb_high, "orb_low": orb_low})
            bull_done = True

        if not bear_done and i >= 2 and h[i] < h[i-1] < h[i-2] and c[i] < orb_high:
            records.append({"day": day, "entry_time": idx[i], "direction": "SHORT",
                             "signal": "LowerHigh", "entry_price": c[i],
                             "orb_high": orb_high, "orb_low": orb_low})
            bear_done = True

        if not bull_done and i >= 2 and lo[i] > lo[i-1] > lo[i-2] and c[i] > orb_low:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                             "signal": "HigherLow", "entry_price": c[i],
                             "orb_high": orb_high, "orb_low": orb_low})
            bull_done = True

print(f"Days: {total_days} total | {skip_range} skipped (range) | {skip_range_day} skipped (range day)")
print(f"Signals generated: {len(records)}")

if not records:
    raise SystemExit("No signals generated — check data or parameters.")

entries_df = pd.DataFrame(records)
entries_df["atr"] = df.loc[entries_df["entry_time"], "atr"].values
n_before = len(entries_df)
entries_df = entries_df.dropna(subset=["atr"]).reset_index(drop=True)
print(f"Dropped {n_before - len(entries_df)} signals with no ATR warmup yet "
      f"(first {ATR_PERIOD + 1} bars of history)")
print(f"Intraday ATR stats at entry — mean: {entries_df['atr'].mean():.1f} pts, "
      f"median: {entries_df['atr'].median():.1f} pts "
      f"(vs. daily-ATR variant's stop distance of ~1.5x150-300+ pts)")


# ── Exit simulators ────────────────────────────────────────────────────────────
def simulate_fixed(entries_df, df):
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


def simulate_atr_trail(entries_df, df):
    out = entries_df.copy()
    out["exit_time"], out["exit_price"], out["pnl_pts"], out["reason"] = pd.NaT, np.nan, np.nan, ""

    for row_i, row in out.iterrows():
        entry_px  = row["entry_price"]
        sign      = 1 if row["direction"] == "SHORT" else -1
        stop_dist = ATR_MULT * row["atr"]
        day_bars  = df[df.index.date == row["day"]]

        best_px    = entry_px
        trail_stop = entry_px + stop_dist if row["direction"] == "SHORT" else entry_px - stop_dist

        for ts, bar in day_bars[day_bars.index >= row["entry_time"]].iterrows():
            if ts == row["entry_time"]:
                continue
            px = bar["close"]
            t  = ts.time()

            if row["direction"] == "SHORT":
                best_px    = min(best_px, px)
                trail_stop = min(trail_stop, best_px + stop_dist)
                hit_stop   = px >= trail_stop
            else:
                best_px    = max(best_px, px)
                trail_stop = max(trail_stop, best_px - stop_dist)
                hit_stop   = px <= trail_stop

            pnl = (entry_px - px) * sign

            if hit_stop:
                reason = "TRAIL_STOP"
            elif t >= HARD_EXIT:
                reason = "HARD_EXIT"
            else:
                continue

            out.at[row_i, "exit_time"], out.at[row_i, "exit_price"] = ts, px
            out.at[row_i, "pnl_pts"], out.at[row_i, "reason"] = pnl, reason
            break

    return out.dropna(subset=["pnl_pts"]).copy()


trades_fixed = simulate_fixed(entries_df, df)
trades_trail = simulate_atr_trail(entries_df, df)

# ── Results ───────────────────────────────────────────────────────────────────
def summarise(df_sub, label):
    if df_sub.empty:
        print(f"\n{label}: no trades")
        return {}
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
    print(f"  Total P&L     : {df_sub['pnl_pts'].sum():+.1f} pts")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  Exits         : {exits}")
    print(f"  Signal types  : {sigs}")
    return {"n": n, "wr": wr, "avg": avg, "total": df_sub["pnl_pts"].sum(), "pf": pf}


print("\n" + "=" * 70)
print("BASELINE — fixed +40pt target / -25pt stop")
print("=" * 70)
res_fixed = summarise(trades_fixed, "ALL TRADES (fixed)")

print("\n" + "=" * 70)
print(f"ATR-TRAILING (intraday 5m ATR) — {ATR_MULT}x ATR({ATR_PERIOD}), no fixed target")
print("=" * 70)
res_trail = summarise(trades_trail, "ALL TRADES (ATR trail, intraday)")

if res_fixed and res_trail:
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  {'Metric':<15}{'Fixed':>12}{'ATR Trail':>12}")
    print(f"  {'Trades':<15}{res_fixed['n']:>12}{res_trail['n']:>12}")
    print(f"  {'Win rate %':<15}{res_fixed['wr']:>12.1f}{res_trail['wr']:>12.1f}")
    print(f"  {'Avg P&L pts':<15}{res_fixed['avg']:>12.2f}{res_trail['avg']:>12.2f}")
    print(f"  {'Total P&L pts':<15}{res_fixed['total']:>12.1f}{res_trail['total']:>12.1f}")
    print(f"  {'Profit factor':<15}{res_fixed['pf']:>12.2f}{res_trail['pf']:>12.2f}")

# ── Save outputs ──────────────────────────────────────────────────────────────
fixed_path = os.path.join(OUT_DIR, "orb_v2_trades_fixed.csv")
trail_path = os.path.join(OUT_DIR, "orb_v2_trades_atr_trail_intraday.csv")
trades_fixed.to_csv(fixed_path, index=False)
trades_trail.to_csv(trail_path, index=False)
print(f"\nTrades saved: {fixed_path}")
print(f"Trades saved: {trail_path}")

for label, trades in (("fixed", trades_fixed), ("atr_trail_intraday", trades_trail)):
    try:
        eq = trades.set_index("exit_time")["pnl_pts"].sort_index()
        eq.index = pd.to_datetime(eq.index).tz_localize(None)
        qs.reports.html(eq, output=os.path.join(OUT_DIR, f"orb_v2_tearsheet_{label}.html"),
                        title=f"ORB v2 — {label} exit variant")
    except Exception as e:
        print(f"Tearsheet skipped ({label}): {e}")

print("\nBacktest complete.")
