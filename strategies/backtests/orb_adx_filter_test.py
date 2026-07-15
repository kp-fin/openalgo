"""
ORB v1 + ORB v2 — ADX Regime Filter Test — VectorBT-style Backtest
Nifty 50 Spot | 2021-07-01 → 2026-06-30

Tests the hypothesis raised in orb.md's own "honest assessment": PF 1.21
is the ceiling of pure parameter tuning for ORB v1; further improvement
"would likely require a regime filter (e.g. trending vs choppy day)".
ADX >= 18 is a standard, cheap proxy for "is this a trending regime."

IMPORTANT CAVEAT — ORB v1 baseline is a REBUILD, not the original:
orb.md states local backtest tooling was retired 2026-07-10 and "no
source file remains" for the script that produced the historical
479-trade / 42.6% WR / PF 1.21 record. This script rebuilds the ORB v1
backtest from the documented canonical spec (15m OR, SHORT-only, 2.5x
target, OR-high stop, 25-75pt range filter, intrabar high/low touches
for stop/target — a more realistic execution simulation than a
close-only check, since real stop/target orders trigger intrabar).
The rebuilt baseline may not exactly reproduce 479/42.6%/1.21 — what's
meaningful here is the WITH-ADX-FILTER vs WITHOUT-ADX-FILTER delta on
the *same* rebuilt methodology, not an exact match to the old number.

ORB v2's baseline reuses the exact entry logic already validated in
orb_v2_backtest.py (close-based checks, matching that script's own
convention) — no rebuild needed there, tooling is intact.

ADX(14) is computed continuously (Wilder RMA, no per-day reset) on each
strategy's own native candle timeframe — 15m for ORB v1, 5m for ORB v2
— so it reflects genuine intraday trend strength with real multi-year
warmup, not a thin same-day sample. Evaluated at the OR-lock bar for
each day (i.e. known before the entry window opens that day — causal,
no lookahead).

P&L in spot points (proxy), consistent with existing ORB backtests.
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

ADX_PERIOD    = 14
ADX_THRESHOLD = 18   # per review recommendation — regime filter cutoff

# ORB v1 canonical params (from indices-system/strategies/orb.md)
V1_RANGE_MIN    = 25
V1_RANGE_MAX    = 75
V1_TARGET_MULT  = 2.5
V1_ENTRY_END    = dtime(12, 0)
V1_HARD_EXIT    = dtime(14, 30)

# ORB v2 canonical params (from orb_v2_backtest.py)
V2_OR_MIN      = 30
V2_OR_MAX      = 150
V2_TARGET_PTS  = 40
V2_STOP_PTS    = 25
V2_ENTRY_END   = dtime(12, 0)
V2_HARD_EXIT   = dtime(14, 30)
V2_RANGE_CHK   = dtime(10, 15)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orb_adx_filter")
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


print(f"Fetching 15m Nifty data {START_DATE} → {END_DATE} (ORB v1) ...")
df15 = _fetch("15m")
print(f"Loaded {len(df15):,} 15m bars: {df15.index[0]} → {df15.index[-1]}")

print(f"Fetching 5m Nifty data {START_DATE} → {END_DATE} (ORB v2) ...")
df5 = _fetch("5m")
print(f"Loaded {len(df5):,} 5m bars: {df5.index[0]} → {df5.index[-1]}")


# ── Wilder RMA + ADX (continuous, no per-day reset, causal) ──────────────────
def wilder_rma(series, period):
    rma = pd.Series(np.nan, index=series.index)
    if len(series) <= period:
        return rma
    rma.iloc[period] = series.iloc[1:period + 1].mean()
    vals, rma_vals = series.values, rma.values
    for i in range(period + 1, len(series)):
        rma_vals[i] = (rma_vals[i - 1] * (period - 1) + vals[i]) / period
    return pd.Series(rma_vals, index=series.index)


def compute_adx(df, period=ADX_PERIOD):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    prev_high, prev_low = high.shift(1), low.shift(1)

    up_move   = high - prev_high
    down_move = prev_low - low
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm   = pd.Series(plus_dm, index=df.index)
    minus_dm  = pd.Series(minus_dm, index=df.index)

    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    atr      = wilder_rma(tr, period)
    plus_di  = 100 * wilder_rma(plus_dm, period) / atr
    minus_di = 100 * wilder_rma(minus_dm, period) / atr
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx      = wilder_rma(dx, period)
    return adx


df15["adx"] = compute_adx(df15)
df5["adx"]  = compute_adx(df5)


# ── ORB v1 signal generation (rebuilt from orb.md canonical spec) ────────────
def generate_orb_v1_signals(df):
    records = []
    total_days, skip_range = 0, 0

    for day, grp in df.groupby(df.index.date):
        total_days += 1
        or_candle = grp.between_time("09:15", "09:15")
        if or_candle.empty:
            continue
        orb_high = or_candle["high"].iloc[0]
        orb_low  = or_candle["low"].iloc[0]
        or_range = orb_high - orb_low

        if or_range < V1_RANGE_MIN or or_range > V1_RANGE_MAX:
            skip_range += 1
            continue

        adx_at_or = grp.between_time("09:15", "09:15")["adx"].iloc[0] if not grp.between_time("09:15", "09:15").empty else np.nan

        window = grp.between_time("09:30", "12:00")
        for ts, bar in window.iterrows():
            if bar["low"] < orb_low:
                records.append({"day": day, "entry_time": ts, "entry_price": orb_low,
                                 "orb_high": orb_high, "orb_low": orb_low,
                                 "or_range": or_range, "adx": adx_at_or})
                break  # max 1 trade/day, first breakout wins

    print(f"ORB v1 — Days: {total_days} | skipped (range): {skip_range} | signals: {len(records)}")
    return pd.DataFrame(records)


def simulate_orb_v1(entries_df, df):
    """Intrabar high/low touches for stop/target — a real breakout order
    (entry at OR low, stop at OR high, target at 2.5x range) triggers
    intrabar, not just on close."""
    out = entries_df.copy()
    out["exit_time"], out["exit_price"], out["pnl_pts"], out["reason"] = pd.NaT, np.nan, np.nan, ""

    for row_i, row in out.iterrows():
        entry_px = row["entry_price"]
        stop_px  = row["orb_high"]
        target_px = entry_px - V1_TARGET_MULT * row["or_range"]
        day_bars = df[df.index.date == row["day"]]

        for ts, bar in day_bars[day_bars.index >= row["entry_time"]].iterrows():
            t = ts.time()
            hi, lo = bar["high"], bar["low"]

            if hi >= stop_px:
                reason, exit_px = "STOP", stop_px
            elif lo <= target_px:
                reason, exit_px = "TARGET", target_px
            elif t >= V1_HARD_EXIT:
                reason, exit_px = "HARD_EXIT", bar["close"]
            else:
                continue

            pnl = entry_px - exit_px  # SHORT: profit when price falls
            out.at[row_i, "exit_time"], out.at[row_i, "exit_price"] = ts, exit_px
            out.at[row_i, "pnl_pts"], out.at[row_i, "reason"] = pnl, reason
            break

    return out.dropna(subset=["pnl_pts"]).copy()


# ── ORB v2 signal generation (identical to orb_v2_backtest.py) ───────────────
def generate_orb_v2_signals(df):
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

        if or_range < V2_OR_MIN or or_range > V2_OR_MAX:
            skip_range += 1
            continue

        at_1015 = grp.between_time("10:15", "10:15")
        if not at_1015.empty:
            c1015 = at_1015["close"].iloc[0]
            if orb_low < c1015 < orb_high:
                skip_range_day += 1
                continue

        adx_at_or = or_window["adx"].iloc[-1] if not or_window.empty else np.nan

        sig = grp.between_time("09:45", "12:00")
        if len(sig) < 2:
            continue

        c, o, h, lo = sig["close"].values, sig["open"].values, sig["high"].values, sig["low"].values
        idx = sig.index
        bear_done = bull_done = False

        for i in range(1, len(sig)):
            t = idx[i].time()
            if t > V2_ENTRY_END:
                break

            if not bear_done and c[i-1] > orb_high and c[i] < orb_high and c[i] < o[i]:
                records.append({"day": day, "entry_time": idx[i], "direction": "SHORT",
                                 "signal": "BearishReject", "entry_price": c[i], "adx": adx_at_or})
                bear_done = True

            if not bull_done and c[i-1] < orb_low and c[i] > orb_low and c[i] > o[i]:
                records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                                 "signal": "BullishReject", "entry_price": c[i], "adx": adx_at_or})
                bull_done = True

            if not bear_done and i >= 2 and h[i] < h[i-1] < h[i-2] and c[i] < orb_high:
                records.append({"day": day, "entry_time": idx[i], "direction": "SHORT",
                                 "signal": "LowerHigh", "entry_price": c[i], "adx": adx_at_or})
                bear_done = True

            if not bull_done and i >= 2 and lo[i] > lo[i-1] > lo[i-2] and c[i] > orb_low:
                records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                                 "signal": "HigherLow", "entry_price": c[i], "adx": adx_at_or})
                bull_done = True

    print(f"ORB v2 — Days: {total_days} | skipped (range): {skip_range} | skipped (range day): {skip_range_day} | signals: {len(records)}")
    return pd.DataFrame(records)


def simulate_orb_v2(entries_df, df):
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

            if pnl >= V2_TARGET_PTS:
                reason = "TARGET"
            elif pnl <= -V2_STOP_PTS:
                reason = "STOP"
            elif t >= V2_HARD_EXIT:
                reason = "HARD_EXIT"
            else:
                continue

            out.at[row_i, "exit_time"], out.at[row_i, "exit_price"] = ts, px
            out.at[row_i, "pnl_pts"], out.at[row_i, "reason"] = pnl, reason
            break

    return out.dropna(subset=["pnl_pts"]).copy()


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
    print(f"\n── {label} ─────────────────────────────────────")
    print(f"  Trades        : {n}")
    print(f"  Win rate      : {wr:.1f}%")
    print(f"  Avg P&L       : {avg:+.2f} pts")
    print(f"  Total P&L     : {df_sub['pnl_pts'].sum():+.1f} pts")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  Exits         : {exits}")
    return {"n": n, "wr": wr, "avg": avg, "total": df_sub["pnl_pts"].sum(), "pf": pf}


def run_strategy(name, entries_df, simulate_fn, df):
    entries_df = entries_df.dropna(subset=["adx"]).reset_index(drop=True)
    unfiltered = simulate_fn(entries_df, df)
    filtered_entries = entries_df[entries_df["adx"] >= ADX_THRESHOLD].reset_index(drop=True)
    filtered = simulate_fn(filtered_entries, df)

    print("\n" + "=" * 70)
    print(f"{name} — ALL SIGNALS (no ADX filter)")
    print("=" * 70)
    res_all = summarise(unfiltered, f"{name} unfiltered")

    print("\n" + "=" * 70)
    print(f"{name} — ADX >= {ADX_THRESHOLD} FILTER ({len(filtered_entries)}/{len(entries_df)} signals pass)")
    print("=" * 70)
    res_adx = summarise(filtered, f"{name} ADX-filtered")

    if res_all and res_adx:
        print(f"\n  {'Metric':<15}{'Unfiltered':>15}{'ADX>=' + str(ADX_THRESHOLD):>15}")
        for metric, fmt in [("n", "{:>15d}"), ("wr", "{:>15.1f}"), ("avg", "{:>15.2f}"),
                             ("total", "{:>15.1f}"), ("pf", "{:>15.2f}")]:
            print(f"  {metric:<15}{fmt.format(res_all[metric])}{fmt.format(res_adx[metric])}")

    unfiltered.to_csv(os.path.join(OUT_DIR, f"{name.lower()}_unfiltered.csv"), index=False)
    filtered.to_csv(os.path.join(OUT_DIR, f"{name.lower()}_adx_filtered.csv"), index=False)

    for label, trades in ((f"{name}_unfiltered", unfiltered), (f"{name}_adx_filtered", filtered)):
        try:
            eq = trades.set_index("exit_time")["pnl_pts"].sort_index()
            eq.index = pd.to_datetime(eq.index).tz_localize(None)
            qs.reports.html(eq, output=os.path.join(OUT_DIR, f"{label.lower()}_tearsheet.html"), title=label)
        except Exception as e:
            print(f"Tearsheet skipped ({label}): {e}")

    return res_all, res_adx


print("\n" + "#" * 70)
print("# ORB v1 (rebuilt from orb.md canonical spec)")
print("#" * 70)
v1_entries = generate_orb_v1_signals(df15)
v1_all, v1_adx = run_strategy("ORB_v1", v1_entries, simulate_orb_v1, df15)

print("\n" + "#" * 70)
print("# ORB v2 (identical logic to orb_v2_backtest.py)")
print("#" * 70)
v2_entries = generate_orb_v2_signals(df5)
v2_all, v2_adx = run_strategy("ORB_v2", v2_entries, simulate_orb_v2, df5)

print("\nBacktest complete.")
