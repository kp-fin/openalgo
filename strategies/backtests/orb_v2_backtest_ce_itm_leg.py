"""
ORB_Spread — CE (Bull Call Spread) Leg: ITM Long vs ATM Long — Structure Test
Nifty 50 Spot | 2021-07-01 -> 2026-06-30

Motivated by a generic "Nifty 50 Bull Call Spread" template Karan surfaced (buys an ITM
call + sells an OTM call) that differs from ORB_Spread's actual live CE leg (buys ATM +
sells OTM1, 50pt width). CE is the strategy's weak/undecided leg (backtest PF 0.99,
pending >=20 CE forward-test trades per the strategy spec's readiness gate) so a genuine
structural alternative is worth testing on its own merits.

This test isolates ONLY the strike-selection question (ITM long vs ATM long, same OTM1
short leg), using real Black-Scholes premium simulation (bs_pricing.py, IV=15% flat,
nearest weekly Tuesday expiry) rather than the intrinsic-value-only proxy used for the
already-adopted 50pt/15%-cost ATM/OTM1 spread -- an ITM long leg carries real time value
that the intrinsic-only model can't represent correctly.

  BASELINE (live design)  : BUY ATM CE  + SELL OTM1 CE   (width = 50pt, one strike)
  CANDIDATE (ITM template): BUY ITM1 CE + SELL OTM1 CE   (width = 100pt, two strikes)

Same entry signals, same spot-points exit triggers (+40 TARGET / -25 STOP / 15:15 HARD),
same single-trade-per-day cap as the current live CE leg -- only the strike selection of
the long leg (and consequent width) changes. Entry/exit spot values come from the existing
spot-points signal+exit simulation (identical logic to the adopted backtest); BS pricing is
applied on top to get real premium P&L for each candidate width.

Caveat: BS pricing here only varies with spot within the day (same trade_date used for
entry and exit -> no intraday time-decay modeled, consistent with bs_pricing.py's existing
day-level convention). No real option-chain data behind either reading -- treat as directional
evidence on the ITM-vs-ATM question, not a precise premium forecast.
"""

import os
import sys
import warnings
from datetime import time as dtime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bs_pricing import bs_price, nearest_atm_strike, nearest_weekly_expiry

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
RANGE_CHK           = dtime(10, 15)
TARGET_PTS, STOP_PTS = 40, 25
STRIKE_STEP         = 50
LOT_SIZE            = 65

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orb_v2_ce_itm_leg")
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

# -- Signal generation: CE (bull) side only, single trade/day (matches live) -------
records = []
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

    sig = grp.between_time("09:45", "12:00")
    if len(sig) < 2:
        continue

    c, o, h, lo = sig["close"].values, sig["open"].values, sig["high"].values, sig["low"].values
    idx = sig.index
    bull_done = False

    for i in range(1, len(sig)):
        if idx[i].time() > ENTRY_END:
            break
        if not bull_done and c[i-1] < orb_low and c[i] > orb_low and c[i] > o[i]:
            records.append({"day": day, "entry_time": idx[i], "signal": "BullishReject", "entry_price": c[i]})
            bull_done = True
        if not bull_done and i >= 2 and lo[i] > lo[i-1] > lo[i-2] and c[i] > orb_low:
            records.append({"day": day, "entry_time": idx[i], "signal": "HigherLow", "entry_price": c[i]})
            bull_done = True

entries_df = pd.DataFrame(records).reset_index(drop=True)
print(f"CE (bull) signals: {len(entries_df)}")


# -- Simulate spot-points exits (identical to adopted baseline) ----------------
def simulate(entries, df):
    out = entries.copy()
    out["exit_time"] = pd.NaT
    out["exit_price"] = np.nan
    out["pnl_pts"] = np.nan
    out["reason"] = ""

    for row_i, row in out.iterrows():
        entry_px = row["entry_price"]
        day_bars = df[df.index.date == row["day"]]

        for ts, bar in day_bars[day_bars.index >= row["entry_time"]].iterrows():
            if ts == row["entry_time"]:
                continue
            px  = bar["close"]
            pnl = px - entry_px  # bull: profits on spot rising
            t   = ts.time()

            if pnl >= TARGET_PTS:
                reason = "TARGET"
            elif pnl <= -STOP_PTS:
                reason = "STOP"
            elif t >= HARD_EXIT:
                reason = "HARD_EXIT"
            else:
                continue

            out.at[row_i, "exit_time"]  = ts
            out.at[row_i, "exit_price"] = px
            out.at[row_i, "pnl_pts"]    = pnl
            out.at[row_i, "reason"]     = reason
            break

    return out.dropna(subset=["pnl_pts"]).copy()


trades = simulate(entries_df, df)
print(f"CE trades simulated (spot-points): {len(trades)}")

# -- Price both structures via Black-Scholes at entry and exit -----------------
rows = []
for _, t in trades.iterrows():
    entry_spot, exit_spot = t["entry_price"], t["exit_price"]
    trade_date = t["day"]
    expiry = nearest_weekly_expiry(trade_date)

    atm = nearest_atm_strike(entry_spot, STRIKE_STEP)
    itm1 = atm - STRIKE_STEP
    otm1 = atm + STRIKE_STEP

    # Baseline: ATM long + OTM1 short
    long_atm_entry = bs_price(entry_spot, atm, trade_date, expiry, "CE")
    short_otm1_entry = bs_price(entry_spot, otm1, trade_date, expiry, "CE")
    net_debit_atm = long_atm_entry - short_otm1_entry

    long_atm_exit = bs_price(exit_spot, atm, trade_date, expiry, "CE")
    short_otm1_exit = bs_price(exit_spot, otm1, trade_date, expiry, "CE")
    net_value_atm_exit = long_atm_exit - short_otm1_exit

    pnl_atm = (net_value_atm_exit - net_debit_atm) * LOT_SIZE

    # Candidate: ITM1 long + OTM1 short (same short leg)
    long_itm1_entry = bs_price(entry_spot, itm1, trade_date, expiry, "CE")
    net_debit_itm = long_itm1_entry - short_otm1_entry

    long_itm1_exit = bs_price(exit_spot, itm1, trade_date, expiry, "CE")
    net_value_itm_exit = long_itm1_exit - short_otm1_exit

    pnl_itm = (net_value_itm_exit - net_debit_itm) * LOT_SIZE

    rows.append({
        "day": trade_date, "signal": t["signal"], "entry_spot": entry_spot, "exit_spot": exit_spot,
        "reason": t["reason"], "pnl_pts_spot": t["pnl_pts"],
        "net_debit_atm": net_debit_atm, "pnl_rupees_atm": pnl_atm,
        "net_debit_itm": net_debit_itm, "pnl_rupees_itm": pnl_itm,
        "width_atm": STRIKE_STEP, "width_itm": STRIKE_STEP * 2,
    })

priced = pd.DataFrame(rows)
priced.to_csv(os.path.join(OUT_DIR, "ce_trades_priced.csv"), index=False)


def summarise(pnl_series, label, cost_series=None):
    n = len(pnl_series)
    wins = (pnl_series > 0).sum()
    wr = wins / n * 100
    avg = pnl_series.mean()
    gw = pnl_series[pnl_series > 0].sum()
    gl = abs(pnl_series[pnl_series <= 0].sum())
    pf = gw / gl if gl > 0 else float("inf")
    print(f"\n-- {label} " + "-" * max(1, 55 - len(label)))
    print(f"  Trades        : {n}")
    print(f"  Win rate      : {wr:.1f}%")
    print(f"  Avg P&L (Rs)  : {avg:+.2f}")
    print(f"  Total P&L (Rs): {pnl_series.sum():+.1f}")
    print(f"  Profit factor : {pf:.2f}")
    if cost_series is not None:
        print(f"  Avg net debit : {cost_series.mean():.2f} pts/unit")
    return {"n": n, "wr": wr, "avg": avg, "total": pnl_series.sum(), "pf": pf}


print("\n" + "=" * 70)
print("CE LEG: ATM/OTM1 (current live design) vs ITM1/OTM1 (candidate)")
print("Black-Scholes premium simulation, IV=15% flat, nearest weekly Tuesday expiry")
print("=" * 70)
atm_stats = summarise(priced["pnl_rupees_atm"], "BASELINE: ATM long + OTM1 short (50pt width)", priced["net_debit_atm"])
itm_stats = summarise(priced["pnl_rupees_itm"], "CANDIDATE: ITM1 long + OTM1 short (100pt width)", priced["net_debit_itm"])

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"  {'Structure':<40} {'N':>6} {'WR%':>7} {'AvgRs':>10} {'PF':>7} {'AvgDebit':>10}")
print("  " + "-" * 82)
for label, stats, debit in [
    ("ATM long + OTM1 short (current)", atm_stats, priced["net_debit_atm"].mean()),
    ("ITM1 long + OTM1 short (candidate)", itm_stats, priced["net_debit_itm"].mean()),
]:
    print(f"  {label:<40} {stats['n']:>6} {stats['wr']:>7.1f} {stats['avg']:>10.2f} {stats['pf']:>7.2f} {debit:>10.2f}")

print("\nCSV saved to:", OUT_DIR)
print("Backtest complete.")
