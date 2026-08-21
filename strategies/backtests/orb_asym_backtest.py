"""
ORB_Asym — asymmetric 1-2-1 butterfly ORB variant (research, 2026-08-21)

New strategy package (not a live ORB_Spread change):
  - Entries: reuse post-pivot LH/HL + OR_MAX<=80 from orb_v2_trades.csv
  - Legs: BUY ATM x1, SELL body x2, BUY far x1 (CE long / PE short)
  - Primary wings: body=150, far=250 (screenshot 150/100)
  - Sensitivity: body=150, far=200 (150/50)
  - Exits re-simulated on 5m NIFTY: target = +body pts, stop = -0.5*debit pts,
    hard exit 15:15. P&L = butterfly intrinsic(delta) - debit.

Does not edit orb_spread_signal.py.
"""

import os
import warnings
from datetime import time as dtime

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")

API_KEY = os.getenv("OPENALGO_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENALGO_API_KEY environment variable before running.")
HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
START_DATE = "2021-07-01"
END_DATE = "2026-06-30"
HARD_EXIT = dtime(15, 15)
IST = pytz.timezone("Asia/Kolkata")

BASE = os.path.dirname(os.path.abspath(__file__))
TRADES_CSV = os.path.join(BASE, "orb_v2", "orb_v2_trades.csv")
OUT_DIR = os.path.join(BASE, "orb_asym")
os.makedirs(OUT_DIR, exist_ok=True)

DEBIT_SPREAD_WIDTH = 50
DEBIT_SPREAD_COST = DEBIT_SPREAD_WIDTH * 0.15  # 7.5
COST_PCTS = (0.20, 0.30, 0.40)
# (body, far, tag)
WING_CFGS = [
    (150, 250, "150/100 screenshot"),
    (150, 200, "150/50 sensitivity"),
]


def round_to_50(px):
    return int(round(px / 50.0) * 50)


def butterfly_intrinsic(delta, body, far):
    return (
        max(delta - 0.0, 0.0)
        - 2.0 * max(delta - body, 0.0)
        + max(delta - far, 0.0)
    )


def metrics(series):
    n = len(series)
    if n == 0:
        return {"n": 0, "wr": float("nan"), "avg": float("nan"),
                "total": 0.0, "pf": float("nan")}
    gw = series[series > 0].sum()
    gl = abs(series[series <= 0].sum())
    return {
        "n": n,
        "wr": (series > 0).mean() * 100,
        "avg": series.mean(),
        "total": series.sum(),
        "pf": gw / gl if gl > 0 else float("inf"),
    }


# ── Entries: post-pivot LH/HL + OR<=80 (do not reuse exit pnl) ───────────────
raw = pd.read_csv(TRADES_CSV, parse_dates=["entry_time", "exit_time"])
raw["or_width"] = raw["orb_high"] - raw["orb_low"]
entries = raw[
    raw["signal"].isin(["HigherLow", "LowerHigh"]) & (raw["or_width"] <= 80)
].copy()
entries["day"] = pd.to_datetime(entries["day"]).dt.date
print(f"Post-pivot entries (LH/HL, OR<=80): {len(entries)}")

# Reference: ORB_Spread debit package on SAME entries, ORIGINAL exits
ref_debit = entries["pnl_pts"].clip(lower=0, upper=DEBIT_SPREAD_WIDTH) - DEBIT_SPREAD_COST
ref_m = metrics(ref_debit)
print(f"Reference ORB_Spread 50pt/15% on original exits: "
      f"n={ref_m['n']} WR={ref_m['wr']:.1f}% PF={ref_m['pf']:.2f} "
      f"total={ref_m['total']:+.1f}")

# ── Fetch 5m NIFTY for exit re-sim ────────────────────────────────────────────
from openalgo import api as openalgo_api

client = openalgo_api(api_key=API_KEY, host=HOST)
print(f"Fetching 5m NIFTY {START_DATE} → {END_DATE} ...")
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


def resim_spot_exits(entries_df, bars, target_pts, stop_pts, hard_exit=HARD_EXIT):
    """Walk 5m bars after entry; spot target/stop/hard exit. Returns spot delta in trade direction."""
    rows = []
    for _, row in entries_df.iterrows():
        entry_px = float(row["entry_price"])
        entry_ts = row["entry_time"]
        if entry_ts.tzinfo is None:
            entry_ts = IST.localize(entry_ts)
        else:
            entry_ts = entry_ts.tz_convert(IST)
        day = row["day"]
        direction = row["direction"]  # LONG=bull CE, SHORT=bear PE
        day_bars = bars[bars.index.date == day]
        day_bars = day_bars[day_bars.index >= entry_ts]

        exit_ts, exit_px, spot_delta, reason = None, None, None, None
        for ts, bar in day_bars.iterrows():
            if ts == entry_ts:
                continue
            px = float(bar["close"])
            # favourable delta in trade direction
            delta = (px - entry_px) if direction == "LONG" else (entry_px - px)
            t = ts.time()
            if delta >= target_pts:
                reason = "TARGET"
            elif delta <= -stop_pts:
                reason = "STOP"
            elif t >= hard_exit:
                reason = "HARD_EXIT"
            else:
                continue
            exit_ts, exit_px, spot_delta = ts, px, delta
            break

        if reason is None and len(day_bars) > 1:
            # last bar of day fallback
            ts = day_bars.index[-1]
            px = float(day_bars.iloc[-1]["close"])
            delta = (px - entry_px) if direction == "LONG" else (entry_px - px)
            exit_ts, exit_px, spot_delta, reason = ts, px, delta, "HARD_EXIT"

        if reason is None:
            continue

        L = round_to_50(entry_px)
        rows.append({
            "day": day,
            "entry_time": entry_ts,
            "direction": direction,
            "signal": row["signal"],
            "entry_price": entry_px,
            "atm_strike": L,
            "exit_time": exit_ts,
            "exit_price": exit_px,
            "spot_delta": spot_delta,
            "reason": reason,
        })
    return pd.DataFrame(rows)


summary_rows = []
print("\n" + "=" * 100)
print("ORB_Asym — body-aware exits + butterfly intrinsic P&L")
print("=" * 100)
print(f"  {'package':<40} {'cost':>5} {'n':>5} {'WR':>7} {'avg':>9} {'total':>10} {'PF':>7}")

# Reference row
print(f"  {'ORB_Spread debit 50/15% (orig exits)':<40} {DEBIT_SPREAD_COST:5.1f} "
      f"{ref_m['n']:5d} {ref_m['wr']:6.1f}% {ref_m['avg']:+8.2f} {ref_m['total']:+9.1f} {ref_m['pf']:7.2f}")
summary_rows.append({
    "package": "ORB_Spread debit 50/15% (orig exits)",
    "body": None, "far": None, "cost_pct": None, "cost_pts": DEBIT_SPREAD_COST,
    **ref_m,
})

primary_trades = None
for body, far, tag in WING_CFGS:
    narrow = min(body, far - body)
    for pct in COST_PCTS:
        debit = narrow * pct
        stop_pts = 0.5 * debit
        # Re-sim exits for this target/stop pair (target always = body)
        sim = resim_spot_exits(entries, df, target_pts=body, stop_pts=stop_pts)
        sim["bf_intrinsic"] = sim["spot_delta"].map(
            lambda d, b=body, f=far: butterfly_intrinsic(d, b, f)
        )
        sim["bf_pnl"] = sim["bf_intrinsic"] - debit
        sim["body"] = body
        sim["far"] = far
        sim["cost_pts"] = debit
        m = metrics(sim["bf_pnl"])
        label = f"ORB_Asym {tag} @ {pct*100:.0f}%"
        print(f"  {label:<40} {debit:5.1f} {m['n']:5d} {m['wr']:6.1f}% "
              f"{m['avg']:+8.2f} {m['total']:+9.1f} {m['pf']:7.2f}")
        summary_rows.append({
            "package": label, "body": body, "far": far,
            "cost_pct": pct, "cost_pts": debit,
            "exits": sim["reason"].value_counts().to_dict(),
            **m,
        })
        # Keep primary (screenshot, 30% cost) trade log
        if body == 150 and far == 250 and abs(pct - 0.30) < 1e-9:
            primary_trades = sim.copy()
            print(f"    exits: {sim['reason'].value_counts().to_dict()}")

summary = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUT_DIR, "orb_asym_summary.csv")
summary.to_csv(summary_csv, index=False)

if primary_trades is not None:
    trades_csv = os.path.join(OUT_DIR, "orb_asym_trades_150_100_cost30.csv")
    primary_trades.to_csv(trades_csv, index=False)
    print(f"\nPrimary trade log -> {trades_csv}")

print(f"Summary -> {summary_csv}")

# Direction split for primary
if primary_trades is not None and not primary_trades.empty:
    print("\nPrimary (150/100, cost 30) by direction:")
    for direction, g in primary_trades.groupby("direction"):
        mm = metrics(g["bf_pnl"])
        print(f"  {direction}: n={mm['n']} WR={mm['wr']:.1f}% PF={mm['pf']:.2f} "
              f"total={mm['total']:+.1f}")

print("\nCaveat: intrinsic-only butterfly proxy; entries shared with ORB LH/HL set but "
      "exits re-simulated for body target. Not live-deployable evidence alone.")
print("Backtest complete.")
