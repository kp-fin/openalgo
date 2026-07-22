"""
ORB_Spread — Target-Composition Test (PF>1.5, WR~50%, Sharpe>=2.0)
Nifty 50 Spot | 2021-07-01 → 2026-06-30

Karan asked (2026-07-22) whether ORB_Spread can realistically reach
PF > 1.5, win rate ~50%, and Sharpe >= 2.0. This does NOT sweep any
parameter (TARGET_PTS/STOP_PTS/OR range/thresholds are all frozen at the
current live config) -- that path is the ORB-v1 "ceiling of pure param
tuning" trap. Instead it tests three PRE-REGISTERED cross-cuts of effects
already validated separately in orb_spread.md, to see if composing them
lands the targets:

  Reference cells (reproduce recorded baselines for context):
    R0  Unfiltered .............. recorded 792 / 44.9% / PF 1.14
    R1  Bear (PE) leg only ...... recorded 393 / 47.1% / PF 1.32
    R2  Quiet prev-day only ..... recorded 397 / 48.4% / PF 1.38  (prev_move <= 0.42%)

  Pre-registered candidate cells:
    A   Bear x Quiet-prev-day
    B   Primary signals only (BearishReject/BullishReject; drop LowerHigh/HigherLow confirmations)
    C   Bear x Quiet x Primary   (full intersection)

PRE-COMMITTED RULE (to resist the garden-of-forking-paths that 8+ prior
filter tests already expose this window to): any cell with < 150 trades
is reported but marked NOT-COUNTABLE -- too thin to treat as evidence,
regardless of how good its PF/WR/Sharpe look. No further cells beyond
these three will be mined; if all three miss the targets, the answer is
"the targets aren't reachable by composing existing validated effects",
not "keep searching."

Two P&L bases reported for every cell, both consistent with existing vault
conventions:
  - SPOT-PROXY (pnl_pts): the basis every recorded ORB_Spread PF/WR uses
    (orb_v2_backtest.py and every filter test). Conservative "does the
    underlying signal have edge" measure. This is the basis the user's
    PF>1.5 / WR~50% targets are comparable against.
  - SPREAD-MODEL: the 50pt/15%-cost debit-spread payoff
    (clamp(pnl_pts,0,50) - 7.5), same as sharpe_baseline.py. The vault
    already flags this as OPTIMISTIC (intrinsic-value-only, no real
    option-chain fills). Sharpe is computed on this basis ONLY because
    that's the only Sharpe convention the vault has -- reported with the
    same Rs 50,000 / 1-lot / 65-qty capital basis as sharpe_baseline.py.

Signal generation is byte-for-byte identical to orb_v2_backtest.py /
orb_v2_backtest_prevday_dow_filter.py. Nothing about entries changes.
"""

import os
import warnings
from datetime import time as dtime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Config (all frozen at current live values -- NOTHING is swept) ────────────
API_KEY = os.getenv("OPENALGO_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENALGO_API_KEY environment variable before running.")
HOST       = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
START_DATE = "2021-07-01"
END_DATE   = "2026-06-30"

OR_MIN, OR_MAX      = 30, 150
ENTRY_END           = dtime(12, 0)
HARD_EXIT           = dtime(15, 15)     # current live value
TARGET_PTS, STOP_PTS = 40, 25
PREV_MOVE_THRESHOLD = 0.42              # current live value, matches recorded 397/395 split

# Spread-model + Sharpe basis (identical to sharpe_baseline.py)
WIDTH        = 50
COST         = WIDTH * 0.15             # 7.5 pts
LOT_SIZE     = 65
ORB_CAPITAL  = 50_000
TRADING_DAYS = 252
MIN_TRADES_COUNTABLE = 150              # pre-committed floor

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orb_v2_target_composition")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Fetch ─────────────────────────────────────────────────────────────────────
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

# ── Lagged prev-day net move (no lookahead) ──────────────────────────────────
daily = df.groupby(df.index.date).agg(open=("open", "first"), close=("close", "last"))
daily["net_move_pct"] = (daily["close"] - daily["open"]).abs() / daily["open"] * 100
daily["prev_net_move_pct"] = daily["net_move_pct"].shift(1)

# ── Signal generation (identical to orb_v2_backtest.py) ──────────────────────
records = []
for day, grp in df.groupby(df.index.date):
    or_window = grp.between_time("09:15", "09:44")
    if len(or_window) < 3:
        continue
    orb_high, orb_low = or_window["high"].max(), or_window["low"].min()
    if not (OR_MIN <= orb_high - orb_low <= OR_MAX):
        continue
    at_1015 = grp.between_time("10:15", "10:15")
    if not at_1015.empty:
        c1015 = at_1015["close"].iloc[0]
        if orb_low < c1015 < orb_high:
            continue
    prev_move = daily["prev_net_move_pct"].get(day, np.nan)
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
                             "signal": "BearishReject", "entry_price": c[i], "prev_move": prev_move})
            bear_done = True
        if not bull_done and c[i-1] < orb_low and c[i] > orb_low and c[i] > o[i]:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                             "signal": "BullishReject", "entry_price": c[i], "prev_move": prev_move})
            bull_done = True
        if not bear_done and i >= 2 and h[i] < h[i-1] < h[i-2] and c[i] < orb_high:
            records.append({"day": day, "entry_time": idx[i], "direction": "SHORT",
                             "signal": "LowerHigh", "entry_price": c[i], "prev_move": prev_move})
            bear_done = True
        if not bull_done and i >= 2 and lo[i] > lo[i-1] > lo[i-2] and c[i] > orb_low:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                             "signal": "HigherLow", "entry_price": c[i], "prev_move": prev_move})
            bull_done = True

entries_df = pd.DataFrame(records)
n_before = len(entries_df)
entries_df = entries_df.dropna(subset=["prev_move"]).reset_index(drop=True)
print(f"Signals: {n_before} | dropped {n_before - len(entries_df)} first-day (no prior reading)")


def simulate(entries):
    out = entries.copy()
    out["exit_time"], out["pnl_pts"], out["reason"] = pd.NaT, np.nan, ""
    for row_i, row in out.iterrows():
        entry_px = row["entry_price"]
        sign     = 1 if row["direction"] == "SHORT" else -1
        day_bars = df[df.index.date == row["day"]]
        for ts, bar in day_bars[day_bars.index >= row["entry_time"]].iterrows():
            if ts == row["entry_time"]:
                continue
            pnl = (entry_px - bar["close"]) * sign
            t   = ts.time()
            if pnl >= TARGET_PTS:
                reason = "TARGET"
            elif pnl <= -STOP_PTS:
                reason = "STOP"
            elif t >= HARD_EXIT:
                reason = "HARD_EXIT"
            else:
                continue
            out.at[row_i, "exit_time"] = ts
            out.at[row_i, "pnl_pts"], out.at[row_i, "reason"] = pnl, reason
            break
    return out.dropna(subset=["pnl_pts"]).copy()


def metrics(trades):
    """Both bases. Spot-proxy PF/WR (recorded-baseline-comparable) + spread-model
    PF/WR + spread-model annualised Sharpe (sharpe_baseline.py convention)."""
    n = len(trades)
    if n == 0:
        return None
    # Spot-proxy
    spot = trades["pnl_pts"]
    wr_spot = (spot > 0).mean() * 100
    gw, gl = spot[spot > 0].sum(), abs(spot[spot <= 0].sum())
    pf_spot = gw / gl if gl > 0 else float("inf")
    # Spread-model
    spread_pts = spot.clip(lower=0, upper=WIDTH) - COST
    wr_spread = (spread_pts > 0).mean() * 100
    gws, gls = spread_pts[spread_pts > 0].sum(), abs(spread_pts[spread_pts <= 0].sum())
    pf_spread = gws / gls if gls > 0 else float("inf")
    # Sharpe on spread-model daily returns
    tmp = trades.copy()
    tmp["exit_date"] = pd.to_datetime(tmp["exit_time"]).dt.date
    tmp["spread_rupees"] = spread_pts.values * LOT_SIZE
    daily_ret = tmp.groupby("exit_date")["spread_rupees"].sum() / ORB_CAPITAL
    sharpe = (daily_ret.mean() / daily_ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
              if daily_ret.std(ddof=1) > 0 else float("nan"))
    return {"n": n, "wr_spot": wr_spot, "pf_spot": pf_spot, "avg_spot": spot.mean(),
            "wr_spread": wr_spread, "pf_spread": pf_spread, "sharpe": sharpe,
            "n_days": len(daily_ret)}


PRIMARY = {"BearishReject", "BullishReject"}

cells = {
    "R0  Unfiltered":                 entries_df,
    "R1  Bear (PE) only":             entries_df[entries_df["direction"] == "SHORT"],
    "R2  Quiet prev-day only":        entries_df[entries_df["prev_move"] <= PREV_MOVE_THRESHOLD],
    "A   Bear x Quiet":               entries_df[(entries_df["direction"] == "SHORT") &
                                                 (entries_df["prev_move"] <= PREV_MOVE_THRESHOLD)],
    "B   Primary signals only":       entries_df[entries_df["signal"].isin(PRIMARY)],
    "C   Bear x Quiet x Primary":     entries_df[(entries_df["direction"] == "SHORT") &
                                                 (entries_df["prev_move"] <= PREV_MOVE_THRESHOLD) &
                                                 (entries_df["signal"].isin(PRIMARY))],
}

rows = []
for label, ent in cells.items():
    m = metrics(simulate(ent.reset_index(drop=True)))
    if m is None:
        continue
    countable = "OK" if m["n"] >= MIN_TRADES_COUNTABLE else f"THIN(<{MIN_TRADES_COUNTABLE})"
    rows.append((label, m, countable))

print("\n" + "=" * 118)
print("ORB_Spread — Target Composition (targets: PF>1.5, WR~50%, Sharpe>=2.0)")
print("=" * 118)
hdr = (f"  {'Cell':<28}{'N':>5}{'WR%(spot)':>11}{'PF(spot)':>10}{'AvgPts':>9}"
       f"{'WR%(sprd)':>11}{'PF(sprd)':>10}{'Sharpe':>9}  {'Countable':>12}")
print(hdr)
print("  " + "-" * 114)
for label, m, countable in rows:
    print(f"  {label:<28}{m['n']:>5}{m['wr_spot']:>11.1f}{m['pf_spot']:>10.2f}{m['avg_spot']:>9.2f}"
          f"{m['wr_spread']:>11.1f}{m['pf_spread']:>10.2f}{m['sharpe']:>9.2f}  {countable:>12}")

print("\n  Targets: PF(spot) > 1.5, WR(spot) ~ 50%, Sharpe >= 2.0")
print("  Sharpe basis: spread-model daily returns / Rs 50,000, sqrt(252) annualised (sharpe_baseline.py).")
print("  Spread-model PF/WR use the OPTIMISTIC 50pt/15%-cost payoff the vault already flags.")
print("  Any THIN cell is NOT evidence regardless of its numbers (pre-committed >=150-trade floor).")

# Save the full-detail trade CSVs for the countable candidate cells
for label, ent in cells.items():
    if not label.startswith(("A", "B", "C")):
        continue
    sim = simulate(ent.reset_index(drop=True))
    fname = label.split()[0] + "_trades.csv"
    sim.to_csv(os.path.join(OUT_DIR, fname), index=False)

print(f"\nCandidate-cell trade CSVs written to {OUT_DIR}")
print("Backtest complete.")
