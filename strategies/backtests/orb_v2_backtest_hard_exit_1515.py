"""
ORB_Spread (entry logic unchanged since ORB v2) — Hard Exit Time Test
Nifty 50 Spot | 2021-07-01 → 2026-06-30

Tests moving HARD_EXIT from 14:30 to 15:15 IST — does giving open
positions ~45 more minutes to resolve naturally (TARGET/STOP) instead
of being force-closed at 14:30 change the statistics?

Same entry logic as orb_v2_backtest.py (identical signal generation,
unaffected by this change since entries only happen 9:45-12:00).
TARGET_PTS (40) and STOP_PTS (25) unchanged — only the hard-exit
cutoff moves. P&L in spot points (proxy), same convention throughout.
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
RANGE_CHK = dtime(10, 15)
TARGET_PTS, STOP_PTS = 40, 25

HARD_EXIT_LEVELS = [dtime(14, 30), dtime(15, 15)]  # baseline vs. the ask

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orb_v2_hard_exit_1515")
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

# ── Signal generation (identical to orb_v2_backtest.py, exit-time-independent) ──
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
print(f"Signals generated: {len(entries_df)} (identical for every hard-exit level tested)")


def simulate(entries_df, df, hard_exit):
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
            elif t >= hard_exit:
                reason = "HARD_EXIT"
            else:
                continue

            out.at[row_i, "exit_time"], out.at[row_i, "exit_price"] = ts, px
            out.at[row_i, "pnl_pts"], out.at[row_i, "reason"] = pnl, reason
            break

    return out.dropna(subset=["pnl_pts"]).copy()


def summarise(df_sub):
    n    = len(df_sub)
    wins = (df_sub["pnl_pts"] > 0).sum()
    wr   = wins / n * 100
    avg  = df_sub["pnl_pts"].mean()
    gw   = df_sub[df_sub["pnl_pts"] > 0]["pnl_pts"].sum()
    gl   = abs(df_sub[df_sub["pnl_pts"] <= 0]["pnl_pts"].sum())
    pf   = gw / gl if gl > 0 else float("inf")
    exits = df_sub["reason"].value_counts().to_dict()
    return {"n": n, "wr": wr, "avg": avg, "total": df_sub["pnl_pts"].sum(), "pf": pf, "exits": exits}


print("\n" + "=" * 100)
print("HARD EXIT TIME COMPARISON — target/stop unchanged at +40/-25 pts")
print("=" * 100)

results = {}
for hard_exit in HARD_EXIT_LEVELS:
    trades = simulate(entries_df, df, hard_exit)
    res = summarise(trades)
    results[hard_exit.strftime("%H:%M")] = res
    trades.to_csv(os.path.join(OUT_DIR, f"orb_v2_hardexit_{hard_exit.strftime('%H%M')}.csv"), index=False)

    print(f"\n── HARD_EXIT = {hard_exit.strftime('%H:%M')} ─────────────────────────────────")
    print(f"  Trades        : {res['n']}")
    print(f"  Win rate      : {res['wr']:.1f}%")
    print(f"  Avg P&L       : {res['avg']:+.2f} pts")
    print(f"  Total P&L     : {res['total']:+.1f} pts")
    print(f"  Profit factor : {res['pf']:.2f}")
    print(f"  Exits         : {res['exits']}")

print("\n" + "=" * 100)
print("COMPARISON")
print("=" * 100)
labels = list(results.keys())
print(f"  {'Metric':<15}" + "".join(f"{l:>15}" for l in labels))
for metric, fmt in [("n", "{:>15d}"), ("wr", "{:>15.1f}"), ("avg", "{:>15.2f}"),
                     ("total", "{:>15.1f}"), ("pf", "{:>15.2f}")]:
    row = f"  {metric:<15}"
    for l in labels:
        row += fmt.format(results[l][metric])
    print(row)

print("\nBacktest complete.")
