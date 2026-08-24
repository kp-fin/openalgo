"""
ORB_Spread — Hold to Expiry Variant, TARGET_PTS=100 (exploratory)
Nifty 50 Spot | 2021-07-01 -> 2026-06-30

Follow-up to orb_v2_backtest_hold_to_expiry.py (target=40, found hold-to-expiry
was a near no-op -- only 1/794 trades ever ran past target/stop) and
orb_v2_backtest_hold_to_expiry_notarget.py (no target at all, found the
adopted 50pt-wide spread's capped payoff made removing the target actively
worse -- PF 4.11 -> 1.24). This tests the middle ground Karan asked for next:
widen the target to +100pts (still finite, still checked continuously) and
give it the room of a full hold-to-expiry window to actually get hit, instead
of being capped by the same-day 15:15 exit.

Entry signal logic (OR/rejection/confirmation signals, range-day filter) is
IDENTICAL to every other ORB_Spread backtest in this vault -- current live
config reproduced exactly (792 trades expected).

Exit logic changes from the current live design:
  - Target widened to +100pt (from +40pt), stop unchanged at -25pt, checked
    on 5m closes -- the walk-forward continues day-by-day (using each day's
    09:15-15:30 5m bars) until that week's expiry, not just same-day.
  - Expiry day is regime-aware: Tuesday from 2025-09-01 onward, Thursday
    before (confirmed switch date, see orb_v2_backtest_dow_expiry_regime_aware.py).
  - If neither target nor stop is hit by expiry, the trade settles at the
    expiry day's last available 5m close before 15:30 (EXPIRY_SETTLE) --
    spot-points intrinsic value, same convention as every other ORB_Spread
    backtest (no historical option chain/IV data exists for this period).

NOTE: the adopted 50pt/15% spread overlay structurally caps payoff at the
50pt width regardless of this target -- a target above 50pts cannot be
captured by that spread. This script still reports the spread-overlay number
for direct comparability, but a +100pt target only matters for the NAKED
spot proxy, or for a future wider-width spread variant (not tested here).

Spread overlay: same adopted 50pt width / 15% cost (7.5pt debit) payoff model
applied on top -- spread_pnl = clamp(pnl_pts, 0, 50) - 7.5 -- so results are
directly comparable to the current PF 3.99 same-day figure. Caveat carried
over unchanged: this ignores that a multi-day hold changes the spread's real
theta exposure (short leg decays favorably, long leg decays unfavorably) --
this script only re-times the SPOT settlement point, it does not attempt to
model time value along the way. Treat as directional exploration, not a
production-ready evidence base.
"""

import os
import warnings
from datetime import date as ddate, time as dtime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("OPENALGO_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENALGO_API_KEY environment variable before running.")
HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
START_DATE = "2021-07-01"
END_DATE = "2026-06-30"

OR_MIN, OR_MAX = 30, 150
ENTRY_END = dtime(12, 0)
RANGE_CHK = dtime(10, 15)
TARGET_PTS, STOP_PTS = 100, 25
EOD_CUTOFF = dtime(15, 30)   # last usable bar of any trading day

EXPIRY_SWITCH_DATE = ddate(2025, 9, 1)   # NIFTY weekly/monthly expiry: Thu -> Tue
OLD_EXPIRY_WEEKDAY = 3   # Thursday (Mon=0 ... Sun=6)
NEW_EXPIRY_WEEKDAY = 1   # Tuesday

SPREAD_WIDTH, SPREAD_COST_PCT = 50, 0.15
SPREAD_COST = SPREAD_WIDTH * SPREAD_COST_PCT

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orb_v2_hold_to_expiry_target100")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Fetch data ────────────────────────────────────────────────────────────────
from openalgo import api as openalgo_api
import pytz

IST = pytz.timezone("Asia/Kolkata")
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

trading_days = sorted(set(df.index.date))
trading_day_set = set(trading_days)


def expiry_weekday_for(day: ddate) -> int:
    return NEW_EXPIRY_WEEKDAY if day >= EXPIRY_SWITCH_DATE else OLD_EXPIRY_WEEKDAY


def expiry_date_for(entry_day: ddate) -> ddate:
    """Nearest trading day matching this regime's expiry weekday, >= entry_day.
    If the calendar expiry weekday isn't itself a trading day (holiday), roll
    forward to the next trading day -- mirrors how OpenAlgo's dynamic /expiry
    endpoint behaves (nearest actual tradable expiry, not a raw calendar date)."""
    wd = expiry_weekday_for(entry_day)
    d = entry_day
    # walk forward to the next (or same) matching weekday
    days_ahead = (wd - d.weekday()) % 7
    candidate = d + pd.Timedelta(days=days_ahead)
    candidate = candidate.date() if hasattr(candidate, "date") else candidate
    # roll forward over holidays until we hit a real trading day
    while candidate not in trading_day_set:
        candidate = candidate + pd.Timedelta(days=1)
        candidate = candidate if isinstance(candidate, ddate) else candidate.date()
        if candidate > entry_day + pd.Timedelta(days=10):
            return None
    return candidate


# ── Signal generation (identical to every other ORB_Spread backtest) ─────────
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
print(f"Signals generated: {len(entries_df)} (expect 792, matching the established baseline)")


# ── Hold-to-expiry simulation ──────────────────────────────────────────────────
def simulate_hold_to_expiry(entries_df, df):
    out = entries_df.copy()
    out["expiry_date"] = out["day"].apply(expiry_date_for)
    dropped = out["expiry_date"].isna().sum()
    if dropped:
        print(f"  Dropping {dropped} signals with no resolvable expiry within 10 days (data-edge cases)")
    out = out.dropna(subset=["expiry_date"]).reset_index(drop=True)

    out["exit_time"], out["exit_price"], out["pnl_pts"], out["reason"] = pd.NaT, np.nan, np.nan, ""
    out["days_held"] = 0

    for row_i, row in out.iterrows():
        entry_px = row["entry_price"]
        sign = 1 if row["direction"] == "SHORT" else -1
        expiry = row["expiry_date"]
        window = df[(df.index.date >= row["day"]) & (df.index.date <= expiry)]
        window = window[window.index >= row["entry_time"]]

        exit_ts, exit_px, reason = None, None, None
        for ts, bar in window.iterrows():
            if ts == row["entry_time"]:
                continue
            px = bar["close"]
            pnl = (entry_px - px) * sign
            d = ts.date()
            t = ts.time()

            if pnl >= TARGET_PTS:
                exit_ts, exit_px, reason = ts, px, "TARGET"
                break
            elif pnl <= -STOP_PTS:
                exit_ts, exit_px, reason = ts, px, "STOP"
                break
            elif d == expiry and t >= EOD_CUTOFF:
                exit_ts, exit_px, reason = ts, px, "EXPIRY_SETTLE"
                break

        if exit_ts is None and not window.empty:
            # fallback: last available bar on/before expiry (holiday-shortened day etc.)
            last = window.iloc[-1]
            exit_ts, exit_px, reason = window.index[-1], last["close"], "EXPIRY_SETTLE"

        if exit_ts is None:
            continue

        pnl = (entry_px - exit_px) * sign
        out.at[row_i, "exit_time"], out.at[row_i, "exit_price"] = exit_ts, exit_px
        out.at[row_i, "pnl_pts"], out.at[row_i, "reason"] = pnl, reason
        out.at[row_i, "days_held"] = (exit_ts.date() - row["day"]).days

    return out.dropna(subset=["pnl_pts"]).copy()


def summarise(df_sub, pnl_col, label):
    if df_sub.empty:
        print(f"\n{label}: no trades")
        return {}
    n = len(df_sub)
    wins = (df_sub[pnl_col] > 0).sum()
    wr = wins / n * 100
    avg = df_sub[pnl_col].mean()
    gw = df_sub[df_sub[pnl_col] > 0][pnl_col].sum()
    gl = abs(df_sub[df_sub[pnl_col] <= 0][pnl_col].sum())
    pf = gw / gl if gl > 0 else float("inf")
    print(f"\n── {label} ─────────────────────────────────────")
    print(f"  Trades        : {n}")
    print(f"  Win rate      : {wr:.1f}%")
    print(f"  Avg P&L       : {avg:+.2f}")
    print(f"  Total P&L     : {df_sub[pnl_col].sum():+.1f}")
    print(f"  Profit factor : {pf:.2f}")
    return {"n": n, "wr": wr, "avg": avg, "total": df_sub[pnl_col].sum(), "pf": pf}


print("\nRunning hold-to-expiry walk-forward (this re-fetches nothing further, but scans the full 5m series per trade -- may take a minute)...")
trades = simulate_hold_to_expiry(entries_df, df)
trades["spread_pnl"] = trades["pnl_pts"].clip(lower=0, upper=SPREAD_WIDTH) - SPREAD_COST
trades.to_csv(os.path.join(OUT_DIR, "hold_to_expiry_trades.csv"), index=False)

print(f"  Exit reasons: {trades['reason'].value_counts().to_dict()}")
print(f"  Days held    : mean {trades['days_held'].mean():.2f}, median {trades['days_held'].median():.0f}, "
      f"max {trades['days_held'].max():.0f}")

print("\n" + "=" * 90)
print(f"HOLD TO EXPIRY, TARGET_PTS=100 — {SPREAD_WIDTH}pt/{SPREAD_COST_PCT*100:.0f}% HEDGED SPREAD ONLY (naked spot suppressed per request)")
print("=" * 90)
res_spread = summarise(trades, "spread_pnl", f"Hold to expiry, target=100 ({SPREAD_WIDTH}pt/{SPREAD_COST_PCT*100:.0f}% spread)")

print("\n" + "=" * 90)
print("COMPARISON — current live (same-day 15:15 hard exit) vs hold-to-expiry target=100 variant, HEDGED SPREAD ONLY")
print("=" * 90)
baseline_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "orb_v2_hard_exit_1515", "orb_v2_hardexit_1515.csv")
if os.path.exists(baseline_csv):
    baseline = pd.read_csv(baseline_csv)
    baseline["spread_pnl"] = baseline["pnl_pts"].clip(lower=0, upper=SPREAD_WIDTH) - SPREAD_COST
    b_spread = summarise(baseline, "spread_pnl", "Same-day 15:15 hard exit (spread)") or {}
else:
    print("  Baseline CSV not found -- skipping direct comparison table.")
    b_spread = {}

print("\nBacktest complete. Output: orb_v2_hold_to_expiry_target100/hold_to_expiry_trades.csv")
