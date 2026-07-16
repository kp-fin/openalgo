"""
ORB_Spread — Previous-Day Behavior & Day-of-Week Filter Test
Nifty 50 Spot | 2021-07-01 → 2026-06-30

Last two regime filters from the original list (CHOP, BB-Width, gap
size already tested -- see orb_spread.md "Signal Filter Testing" /
Karan's earlier conversation, CHOP rejected, BB-Width promising, gap
inconclusive). Combined into one script since both are cheap to
compute from data already being fetched for entries -- no new
indicator warmup needed.

1) PREVIOUS-DAY NET MOVE — does a big directional trend day yesterday
   predict anything about today's fade-friendliness? net_move_pct =
   |yesterday close - yesterday open| / yesterday open. Lagged by one
   day (today's signals use YESTERDAY's reading, never today's own) --
   no lookahead. No strong prior on direction, tested both ways
   (filter for low vs high prior-day net move).

2) DAY OF WEEK / THURSDAY EXPIRY — weekly NIFTY options expire
   Thursday; pin risk / unusual price action near max-pain is a common
   claim. Simple categorical split using each entry's own weekday, no
   indicator computation needed.

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
HARD_EXIT = dtime(15, 15)
RANGE_CHK = dtime(10, 15)
TARGET_PTS, STOP_PTS = 40, 25

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orb_v2_prevday_dow_filter")
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

# ── Daily OHLC derived from 5m bars, then lagged net-move% (no lookahead) ────
daily = df.groupby(df.index.date).agg(open=("open", "first"), close=("close", "last"))
daily["net_move_pct"] = (daily["close"] - daily["open"]).abs() / daily["open"] * 100
daily["prev_net_move_pct"] = daily["net_move_pct"].shift(1)   # yesterday's reading, used today
print(f"Prev-day net move %% stats: mean={daily['prev_net_move_pct'].mean():.3f}%, "
      f"median={daily['prev_net_move_pct'].median():.3f}%")

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

    prev_move = daily["prev_net_move_pct"].get(day, np.nan)
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
                             "prev_move": prev_move, "dow": dow})
            bear_done = True
        if not bull_done and c[i-1] < orb_low and c[i] > orb_low and c[i] > o[i]:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                             "signal": "BullishReject", "entry_price": c[i],
                             "prev_move": prev_move, "dow": dow})
            bull_done = True
        if not bear_done and i >= 2 and h[i] < h[i-1] < h[i-2] and c[i] < orb_high:
            records.append({"day": day, "entry_time": idx[i], "direction": "SHORT",
                             "signal": "LowerHigh", "entry_price": c[i],
                             "prev_move": prev_move, "dow": dow})
            bear_done = True
        if not bull_done and i >= 2 and lo[i] > lo[i-1] > lo[i-2] and c[i] > orb_low:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                             "signal": "HigherLow", "entry_price": c[i],
                             "prev_move": prev_move, "dow": dow})
            bull_done = True

entries_df = pd.DataFrame(records)
n_before = len(entries_df)
entries_df = entries_df.dropna(subset=["prev_move"]).reset_index(drop=True)
print(f"Signals generated: {n_before} | dropped {n_before - len(entries_df)} with no prior-day reading (first day)")
print(f"Day-of-week counts:\n{entries_df['dow'].value_counts()}")


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
print("PART 1: PREVIOUS-DAY NET MOVE %")
print("=" * 90)
res_all = summarise(all_trades, "Unfiltered")

med = entries_df["prev_move"].median()
low_move_entries = entries_df[entries_df["prev_move"] <= med].reset_index(drop=True)
high_move_entries = entries_df[entries_df["prev_move"] > med].reset_index(drop=True)
res_low = summarise(simulate(low_move_entries, df), f"Prev-day move <= median ({med:.2f}%)")
res_high = summarise(simulate(high_move_entries, df), f"Prev-day move > median ({med:.2f}%)")

print("\n" + "=" * 90)
print("PART 2: DAY OF WEEK")
print("=" * 90)
dow_results = {}
for dow in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
    dow_entries = entries_df[entries_df["dow"] == dow].reset_index(drop=True)
    if dow_entries.empty:
        continue
    dow_results[dow] = summarise(simulate(dow_entries, df), dow)

print("\n" + "=" * 90)
print("COMPARISON — Previous-day move")
print("=" * 90)
labels1 = ["Unfiltered", "Low prev-move", "High prev-move"]
res1 = [res_all, res_low, res_high]
print(f"  {'Metric':<15}" + "".join(f"{l:>16}" for l in labels1))
for metric, fmt in [("n", "{:>16d}"), ("wr", "{:>16.1f}"), ("avg", "{:>16.2f}"), ("pf", "{:>16.2f}")]:
    row = f"  {metric:<15}"
    for r in res1:
        row += fmt.format(r[metric]) if r else f"{'—':>16}"
    print(row)

print("\n" + "=" * 90)
print("COMPARISON — Day of week")
print("=" * 90)
print(f"  {'Day':<12}{'Trades':>8}{'WR%':>8}{'Avg':>8}{'PF':>8}")
for dow, res in dow_results.items():
    if res:
        print(f"  {dow:<12}{res['n']:>8}{res['wr']:>8.1f}{res['avg']:>8.2f}{res['pf']:>8.2f}")

print("\nBacktest complete.")
