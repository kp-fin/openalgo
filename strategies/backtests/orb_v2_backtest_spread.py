"""
ORB v2 — Debit Spread P&L Overlay (Bull Call Spread / Bear Put Spread)
Nifty 50 Spot | 2021-07-01 → 2026-06-30

Tests how ORB v2's statistics change if LONG signals (currently modeled as
a naked ATM CE) are traded as a Bull Call Spread, and SHORT signals
(naked ATM PE) as a Bear Put Spread, instead.

WIDTH = 100 pts, COST = 35% of width (35 pts) — both set per user direction
(2026-07-15). No historical NIFTY option chain/IV data exists for this
period (orb_v2.md's own Backtest Plan already notes the naked-long P&L
is a spot-points proxy, not real premium data) — a spread model needs
these two extra assumptions on top of that same limitation, and neither
can be derived empirically here.

KEY STRUCTURAL POINT: because WIDTH (100) > TARGET_PTS (40), the
underlying entry/exit timing is UNCHANGED from the naked-long baseline —
the strategy's own fixed target/stop already caps gains before the
spread's own 100pt cap would ever bind. So this script does NOT re-run
entry/exit simulation; it reuses the exact trade-level results already
saved from the baseline backtest (orb_v2_atr_variants/orb_v2_trades_fixed.csv,
792 trades, confirmed to exactly match orb_v2_backtest.py's original
792/44.9%/1.14 result) and re-derives spread P&L from each trade's
already-computed pnl_pts.

PAYOFF MODEL (intrinsic-value-only — read this before trusting the
numbers below):
    spread_value_at_exit = clamp(pnl_pts, 0, WIDTH)
    spread_pnl            = spread_value_at_exit - COST

This treats the spread as if valued at its EXPIRY payoff (pure intrinsic
value) at the moment of exit, with ZERO extrinsic/time value remaining.
That is almost certainly too pessimistic for real trades: these are
same-day intraday exits against a *weekly* expiry, so on most days
(everything except Thursday 0DTE) a stopped-out or modestly-profitable
spread would still retain real time value at exit, not be worth exactly
its intrinsic value. The practical effect: EVERY naked-losing trade
(pnl_pts <= 0) collapses to the same -35pt (full premium) loss under
this model, and EVERY naked-winning trade below +35pts (the cost) shows
as a net LOSS on the spread even though the underlying move was
favorable. Treat the numbers below as a LOWER BOUND / pessimistic case
for the spread's real-world performance, not a precise estimate — a
more realistic model would need actual IV/theta data this backtest
period doesn't have.
"""

import os

import pandas as pd

WIDTH    = 100
COST_PCT = 0.35
COST     = WIDTH * COST_PCT

TRADES_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "orb_v2_atr_variants", "orb_v2_trades_fixed.csv")

trades = pd.read_csv(TRADES_CSV)
print(f"Loaded {len(trades)} baseline trades from {TRADES_CSV}")
print(f"Spread assumptions: width={WIDTH}pts, cost={COST_PCT*100:.0f}% of width = {COST:.1f}pts\n")

trades["spread_value"] = trades["pnl_pts"].clip(lower=0, upper=WIDTH)
trades["spread_pnl"]   = trades["spread_value"] - COST


def summarise(df_sub, pnl_col, label):
    if df_sub.empty:
        print(f"\n{label}: no trades")
        return {}
    n    = len(df_sub)
    wins = (df_sub[pnl_col] > 0).sum()
    wr   = wins / n * 100
    avg  = df_sub[pnl_col].mean()
    gw   = df_sub[df_sub[pnl_col] > 0][pnl_col].sum()
    gl   = abs(df_sub[df_sub[pnl_col] <= 0][pnl_col].sum())
    pf   = gw / gl if gl > 0 else float("inf")
    print(f"\n── {label} ─────────────────────────────────────")
    print(f"  Trades        : {n}")
    print(f"  Win rate      : {wr:.1f}%")
    print(f"  Avg P&L       : {avg:+.2f} pts")
    print(f"  Total P&L     : {df_sub[pnl_col].sum():+.1f} pts")
    print(f"  Profit factor : {pf:.2f}")
    return {"n": n, "wr": wr, "avg": avg, "total": df_sub[pnl_col].sum(), "pf": pf}


bull = trades[trades["direction"] == "LONG"]   # bull call spread
bear = trades[trades["direction"] == "SHORT"]  # bear put spread

print("=" * 78)
print("NAKED LONG (existing proxy, unchanged baseline)")
print("=" * 78)
naked_bull = summarise(bull, "pnl_pts", "Bullish (naked ATM CE)")
naked_bear = summarise(bear, "pnl_pts", "Bearish (naked ATM PE)")
naked_all  = summarise(trades, "pnl_pts", "ALL (naked)")

print("\n" + "=" * 78)
print(f"SPREAD MODEL — width={WIDTH}pts, cost={COST:.0f}pts (intrinsic-only, pessimistic on losers)")
print("=" * 78)
spread_bull = summarise(bull, "spread_pnl", "Bull Call Spread")
spread_bear = summarise(bear, "spread_pnl", "Bear Put Spread")
spread_all  = summarise(trades, "spread_pnl", "ALL (spread)")

print("\n" + "=" * 78)
print("COMPARISON")
print("=" * 78)
rows = [
    ("Bullish", naked_bull, spread_bull),
    ("Bearish", naked_bear, spread_bear),
    ("Combined", naked_all, spread_all),
]
print(f"  {'':<10}{'':>4}{'Naked WR':>10}{'Spread WR':>11}{'Naked Avg':>11}{'Spread Avg':>12}{'Naked PF':>10}{'Spread PF':>11}")
for label, nk, sp in rows:
    if not nk or not sp:
        continue
    print(f"  {label:<10}{nk['n']:>4}{nk['wr']:>9.1f}%{sp['wr']:>10.1f}%"
          f"{nk['avg']:>11.2f}{sp['avg']:>12.2f}{nk['pf']:>10.2f}{sp['pf']:>11.2f}")

# ── Breakeven analysis: what naked pt-gain is needed for the spread to breakeven ──
print("\n" + "=" * 78)
print("BREAKEVEN CHECK")
print("=" * 78)
print(f"  Spread breaks even at naked pnl_pts >= {COST:.0f} (cost). Strategy's own TARGET_PTS = 40.")
print(f"  A TARGET-hit trade (+40pts naked) nets only +{40 - COST:.0f}pts on the spread —"
      f" {(40-COST)/40*100:.0f}% of the naked gain, because cost ({COST:.0f}) eats most of the capped upside (width {WIDTH}, but exit still capped at 40 by the strategy's own target).")
below_cost = (trades["pnl_pts"] > 0) & (trades["pnl_pts"] < COST)
print(f"  Naked-winning trades that still show a SPREAD loss (0 < pnl_pts < {COST:.0f}): "
      f"{below_cost.sum()} of {len(trades)} ({below_cost.sum()/len(trades)*100:.1f}%)")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "orb_v2_atr_variants", "orb_v2_trades_spread_model.csv")
trades.to_csv(out_path, index=False)
print(f"\nTrade-level detail saved: {out_path}")
