"""
ORB_Spread — CE (Bull Call Spread) Leg: Re-entry After Exit — Structure Test
Nifty 50 Spot | 2021-07-01 -> 2026-06-30

Motivated by a generic "Nifty 50 Bull Call Spread" template Karan surfaced that
re-enters the same leg structure after an exit. ORB_Spread's live design caps at
ONE CE trade/day (`bull_traded` flag in orb_spread_signal.py) -- multi-entry has
never been tested. CE is the strategy's weak/undecided leg (backtest PF 0.99,
pending >=20 CE forward-test trades) so this is tested on its own merits.

  BASELINE (live design): first valid CE signal wins, no re-entry same day
  CANDIDATE             : after a CE position exits (TARGET/STOP -- HARD_EXIT
                          ends the day), resume scanning for a new CE signal;
                          re-enter if one fires before the 12:00 entry cutoff.

Both variants use one integrated per-day walk-forward loop (entry-check and
exit-check interleaved on the same 5m bar sequence, mirroring the live script's
actual event loop) so the only difference between them is the re-entry gate
itself -- same signal logic, same +40/-25/15:15 exit rules, same day-skip range
filter. Spread economics applied on top via the already-adopted 50pt/15%-cost
intrinsic-value proxy (clamp(pnl_pts, 0, WIDTH) - COST), consistent with the
rest of this strategy's spread-modeling evidence.
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
WIDTH, COST_PCT      = 50, 0.15
COST                 = WIDTH * COST_PCT

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orb_v2_ce_reentry")
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


def walk_day(day_walk, allow_reentry):
    """Single integrated entry+exit walk over one day's post-OR-lock bars.
    Returns list of trade dicts. Only CE (bull) signals."""
    idx = day_walk.index
    c, o, lo = day_walk["close"].values, day_walk["open"].values, day_walk["low"].values
    trades = []
    in_position = False
    already_traded = False
    entry_price = entry_time = entry_signal = None

    for i in range(len(day_walk)):
        t = idx[i].time()

        if in_position:
            pnl = c[i] - entry_price
            reason = None
            if pnl >= TARGET_PTS:
                reason = "TARGET"
            elif pnl <= -STOP_PTS:
                reason = "STOP"
            elif t >= HARD_EXIT:
                reason = "HARD_EXIT"
            if reason:
                trades.append({"entry_time": entry_time, "entry_price": entry_price,
                               "exit_time": idx[i], "exit_price": c[i], "pnl_pts": pnl,
                               "reason": reason, "signal": entry_signal})
                in_position = False
                already_traded = True
            continue

        if t > ENTRY_END:
            continue
        if already_traded and not allow_reentry:
            continue

        sig_fired = None
        if i >= 1 and c[i-1] < orb_low and c[i] > orb_low and c[i] > o[i]:
            sig_fired = "BullishReject"
        elif i >= 2 and lo[i] > lo[i-1] > lo[i-2] and c[i] > orb_low:
            sig_fired = "HigherLow"

        if sig_fired:
            in_position = True
            entry_price, entry_time, entry_signal = c[i], idx[i], sig_fired

    return trades


all_baseline, all_candidate = [], []

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

    day_walk = grp.between_time("09:45", "15:25")
    if len(day_walk) < 2:
        continue

    base_trades = walk_day(day_walk, allow_reentry=False)
    cand_trades = walk_day(day_walk, allow_reentry=True)

    for tr in base_trades:
        tr["day"] = day
        all_baseline.append(tr)
    for tr in cand_trades:
        tr["day"] = day
        all_candidate.append(tr)

baseline_df  = pd.DataFrame(all_baseline)
candidate_df = pd.DataFrame(all_candidate)

baseline_df["spread_pnl"]  = baseline_df["pnl_pts"].clip(lower=0, upper=WIDTH) - COST
candidate_df["spread_pnl"] = candidate_df["pnl_pts"].clip(lower=0, upper=WIDTH) - COST

baseline_df.to_csv(os.path.join(OUT_DIR, "baseline_single_entry_trades.csv"), index=False)
candidate_df.to_csv(os.path.join(OUT_DIR, "candidate_reentry_trades.csv"), index=False)


def summarise(df_sub, pnl_col, label):
    n = len(df_sub)
    wins = (df_sub[pnl_col] > 0).sum()
    wr = wins / n * 100
    avg = df_sub[pnl_col].mean()
    gw = df_sub[df_sub[pnl_col] > 0][pnl_col].sum()
    gl = abs(df_sub[df_sub[pnl_col] <= 0][pnl_col].sum())
    pf = gw / gl if gl > 0 else float("inf")
    print(f"\n-- {label} " + "-" * max(1, 55 - len(label)))
    print(f"  Trades        : {n}")
    print(f"  Win rate      : {wr:.1f}%")
    print(f"  Avg           : {avg:+.2f}")
    print(f"  Total         : {df_sub[pnl_col].sum():+.1f}")
    print(f"  Profit factor : {pf:.2f}")
    return {"n": n, "wr": wr, "avg": avg, "total": df_sub[pnl_col].sum(), "pf": pf}


print("\n" + "=" * 70)
print("CE LEG: single-entry/day (current live) vs re-entry-after-exit (candidate)")
print("=" * 70)
print("\n--- Spot-points basis ---")
summarise(baseline_df, "pnl_pts", "BASELINE (single entry/day) — spot pts")
summarise(candidate_df, "pnl_pts", "CANDIDATE (re-entry after exit) — spot pts")

print("\n--- Spread economics (50pt width, 15% cost) ---")
b_spread = summarise(baseline_df, "spread_pnl", "BASELINE (single entry/day) — spread")
c_spread = summarise(candidate_df, "spread_pnl", "CANDIDATE (re-entry after exit) — spread")

trades_per_day_base = baseline_df.groupby("day").size()
trades_per_day_cand = candidate_df.groupby("day").size()
print(f"\nTrades/day — baseline: mean {trades_per_day_base.mean():.2f}, max {trades_per_day_base.max()}")
print(f"Trades/day — candidate: mean {trades_per_day_cand.mean():.2f}, max {trades_per_day_cand.max()}")
print(f"Days with >=2 candidate trades: {(trades_per_day_cand >= 2).sum()} of {trades_per_day_cand.shape[0]} traded days")

print("\n" + "=" * 70)
print("SUMMARY TABLE (spread economics, 50pt/15% cost)")
print("=" * 70)
print(f"  {'Rule':<35} {'N':>6} {'WR%':>7} {'Avg':>8} {'Total':>10} {'PF':>7}")
print("  " + "-" * 76)
for label, stats in [("Baseline (single entry/day)", b_spread), ("Candidate (re-entry after exit)", c_spread)]:
    print(f"  {label:<35} {stats['n']:>6} {stats['wr']:>7.1f} {stats['avg']:>8.2f} {stats['total']:>10.1f} {stats['pf']:>7.2f}")

print("\nCSVs saved to:", OUT_DIR)
print("Backtest complete.")
