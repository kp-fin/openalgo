"""
ORB v2 — 150pt / 200pt Width Gap-Fill
Nifty 50 Spot | 2021-07-01 -> 2026-06-30

Fills the gap flagged after orb_v2_backtest_spread.py (100pt/35%, PF 0.30,
rejected) and orb_v2_backtest_spread_narrow.py (50pt/15%, PF 3.99, adopted):
those two scripts never tested 150pt or 200pt widths. This reuses the same
792-trade naked-proxy set and payoff model, adding 150/200pt rows at the
same three cost tiers (15%, 20%, 35%) already used across the sweep, plus
100pt and 50pt/15% as reference rows for continuity.

Same caveat as every script in this family: pnl_pts is a spot-points proxy
with no real NIFTY option chain/IV data behind it. Spread value at exit =
clamp(pnl_pts, 0, WIDTH) - COST, intrinsic-only. Treat PF as a lower bound
and the relative comparison across widths as the informative part, not
these numbers as a precise real-money forecast.
"""

import os

import pandas as pd

TRADES_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "orb_v2_spread_wide_target", "orb_v2_target40.csv")
trades = pd.read_csv(TRADES_CSV)
print(f"Loaded {len(trades)} baseline trades (unchanged entries/exits — target=40)\n")

CONFIGS = [
    (50,  0.15),  # adopted config — reference
    (100, 0.35),  # prior rejected config — reference
    (150, 0.15),
    (150, 0.20),
    (150, 0.35),
    (200, 0.15),
    (200, 0.20),
    (200, 0.35),
]


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


print("=" * 100)
print("WIDTH / COST SWEEP — 150pt/200pt gap-fill, target unchanged at 40pts, stop unchanged at 25pts")
print("=" * 100)
print(f"  {'Width':<8}{'Cost%':<8}{'Cost pts':<10}{'WR':>8}{'Avg P&L':>11}{'Total P&L':>12}{'PF':>8}")

naked = summarise(trades, "pnl_pts")
print(f"  {'(naked)':<8}{'—':<8}{'—':<10}{naked['wr']:>7.1f}%{naked['avg']:>11.2f}{naked['total']:>12.1f}{naked['pf']:>8.2f}")
print("  " + "-" * 90)

for width, cost_pct in CONFIGS:
    cost = width * cost_pct
    trades["spread_pnl"] = trades["pnl_pts"].clip(lower=0, upper=width) - cost
    res = summarise(trades, "spread_pnl")
    print(f"  {width:<8}{cost_pct*100:.0f}%{'':<5}{cost:<10.1f}{res['wr']:>7.1f}%"
          f"{res['avg']:>11.2f}{res['total']:>12.1f}{res['pf']:>8.2f}")

print("\nNote: cost pts = width * cost_pct, subtracted from clamped payoff. Reference rows")
print("(50pt/15% adopted, 100pt/35% rejected) are unchanged from prior scripts — same CSV, same model.")
print("Backtest complete.")
