"""
ORB v2 — Narrower/Cheaper Spread Sweep
Nifty 50 Spot | 2021-07-01 → 2026-06-30

Follow-up to orb_v2_backtest_spread.py (100pt width / 35% cost, PF 0.30)
and orb_v2_backtest_spread_wide_target.py (widening the target only got
to PF 0.53 at best, still unprofitable). This tests the other lever:
a narrower AND cheaper spread instead of a wider target.

All widths tested here (30, 50) stay >= the strategy's existing 40pt
target, so entries/exits are UNCHANGED from the naked-long baseline —
reuses the same 792-trade set as before (orb_v2_atr_variants/orb_v2_trades_fixed.csv),
no re-fetch needed. 100pt/35% included as a reference row.

Same payoff model and same caveat as the prior two scripts: spread value
at exit = clamp(pnl_pts, 0, WIDTH) - COST, intrinsic-only (pessimistic on
losers — real losses on non-Thursday intraday exits would likely be
smaller than the full premium). Treat PF as a lower bound, not a precise
estimate — the relative comparison across configs is what's informative.
"""

import os

import pandas as pd

TRADES_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "orb_v2_atr_variants", "orb_v2_trades_fixed.csv")
trades = pd.read_csv(TRADES_CSV)
print(f"Loaded {len(trades)} baseline trades (unchanged entries/exits — target=40, all tested widths >= 40)\n")

CONFIGS = [
    (100, 0.35),  # reference — already tested
    (50,  0.20),
    (50,  0.15),
    (30,  0.20),
    (30,  0.15),
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
print("WIDTH / COST SWEEP — combined (bull + bear), target unchanged at 40pts, stop unchanged at 25pts")
print("=" * 100)
print(f"  {'Width':<8}{'Cost%':<8}{'Cost pts':<10}{'Breakeven':<11}{'WR':>8}{'Avg P&L':>11}{'Total P&L':>12}{'PF':>8}")

naked = summarise(trades, "pnl_pts")
print(f"  {'(naked)':<8}{'—':<8}{'—':<10}{'—':<11}{naked['wr']:>7.1f}%{naked['avg']:>11.2f}{naked['total']:>12.1f}{naked['pf']:>8.2f}")
print("  " + "-" * 90)

for width, cost_pct in CONFIGS:
    cost = width * cost_pct
    trades["spread_pnl"] = trades["pnl_pts"].clip(lower=0, upper=width) - cost
    res = summarise(trades, "spread_pnl")
    print(f"  {width:<8}{cost_pct*100:.0f}%{'':<5}{cost:<10.1f}{cost:<11.1f}{res['wr']:>7.1f}%"
          f"{res['avg']:>11.2f}{res['total']:>12.1f}{res['pf']:>8.2f}")

print("\nNote: breakeven = naked pnl_pts needed for spread_pnl >= 0 = cost (since spread rarely caps, target=40 < most widths tested).")
print("Backtest complete.")
