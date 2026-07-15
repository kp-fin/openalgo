"""
ORB v2 — Wider Target Calibrated for Spread Structure
Nifty 50 Spot | 2021-07-01 → 2026-06-30

Follow-up to orb_v2_backtest_spread.py, which found that under the
strategy's existing 40pt fixed target, a 100pt-wide / 35pt-cost debit
spread nets only +5pts on a winning trade (40 - 35) while a stopped-out
trade loses the full 35pt premium under the intrinsic-only payoff model
— a target built for a naked long leaves almost no margin once a spread's
cost is subtracted.

This script re-simulates ORB v2's EXIT logic (entries are unchanged —
signal generation doesn't depend on target) at several TARGET_PTS levels
between the original 40 and the spread's own 100pt cap, to see whether a
wider target — giving the position more room to run before exiting,
consistent with a spread's higher breakeven — improves the spread-model
economics. STOP_PTS (25) is left unchanged; only the target widens.

Same payoff model and same caveat as orb_v2_backtest_spread.py: spread
value at exit = clamp(pnl_pts, 0, WIDTH) - COST, which assumes zero
extrinsic value survives to exit (pessimistic on losers — real losses on
non-Thursday intraday exits would likely be smaller than the full
premium). Treat PF here as a lower bound, not a precise estimate — but
the RELATIVE comparison across target levels is still informative.
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
HARD_EXIT = dtime(14, 30)
RANGE_CHK = dtime(10, 15)
STOP_PTS  = 25

WIDTH    = 100
COST_PCT = 0.35
COST     = WIDTH * COST_PCT

TARGET_LEVELS = [40, 60, 80, 100]  # 40 = original, for reference

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orb_v2_spread_wide_target")
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

# ── Signal generation (identical to orb_v2_backtest.py, target-independent) ──
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
print(f"Signals generated: {len(entries_df)} (target-independent, same for every level tested)")


def simulate(entries_df, df, target_pts):
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

            if pnl >= target_pts:
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


def summarise(df_sub, pnl_col):
    if df_sub.empty:
        return {"n": 0, "wr": 0, "avg": 0, "total": 0, "pf": 0}
    n    = len(df_sub)
    wins = (df_sub[pnl_col] > 0).sum()
    wr   = wins / n * 100
    avg  = df_sub[pnl_col].mean()
    gw   = df_sub[df_sub[pnl_col] > 0][pnl_col].sum()
    gl   = abs(df_sub[df_sub[pnl_col] <= 0][pnl_col].sum())
    pf   = gw / gl if gl > 0 else float("inf")
    return {"n": n, "wr": wr, "avg": avg, "total": df_sub[pnl_col].sum(), "pf": pf}


print("\n" + "=" * 100)
print(f"TARGET LEVEL SWEEP — spread width={WIDTH}pts, cost={COST:.0f}pts, stop unchanged at {STOP_PTS}pts")
print("=" * 100)
print(f"  {'Target':<8}{'Trades':>8}{'Naked WR':>10}{'Naked Avg':>11}{'Naked PF':>10}"
      f"{'  |  Spread WR':>14}{'Spread Avg':>12}{'Spread PF':>11}{'Exit mix':>30}")

results = []
for target in TARGET_LEVELS:
    trades = simulate(entries_df, df, target)
    trades["spread_pnl"] = trades["pnl_pts"].clip(lower=0, upper=WIDTH) - COST

    naked  = summarise(trades, "pnl_pts")
    spread = summarise(trades, "spread_pnl")
    exits  = trades["reason"].value_counts().to_dict()

    spread_wr_str = f"  |  {spread['wr']:.1f}%"
    print(f"  {target:<8}{naked['n']:>8}{naked['wr']:>9.1f}%{naked['avg']:>11.2f}{naked['pf']:>10.2f}"
          f"{spread_wr_str:>14}{spread['avg']:>12.2f}{spread['pf']:>11.2f}{str(exits):>30}")

    results.append({"target": target, "naked": naked, "spread": spread, "exits": exits})
    trades.to_csv(os.path.join(OUT_DIR, f"orb_v2_target{target}.csv"), index=False)

print("\nNote: TARGET=40 row reproduces the original baseline/spread-overlay result as a sanity check.")
print("Backtest complete.")
