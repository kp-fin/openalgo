"""
Sharpe Ratio Baseline — ORB_Spread and EMA Regime Crossover
Goal (Karan, 2026-07-18): Sharpe ratio >= 2 for both live-trading strategies.
Not a promotion/readiness gate — a live-trading performance target, computed
here against each strategy's existing backtest evidence to establish where
each currently stands.

Methodology:
- Daily portfolio return series -> annualised Sharpe = mean(daily) / std(daily) * sqrt(252).
  Standard convention, risk-free rate treated as 0 (not a meaningful adjustment
  at these return magnitudes).

ORB_Spread: source trades are the 792-trade naked-proxy backtest (spot points,
orb_v2/orb_v2_trades.csv), reused throughout orb_spread.md for every downstream
variant. Applies the same debit-spread payoff formula already adopted live
(50pt width / 15% cost -> COST=7.5pts): spread_pnl = clamp(pnl_pts, 0, WIDTH) - COST.

Capital basis confirmed by Karan 2026-07-18: Rs 50,000 real account allocated
to this strategy, fixed 1 lot per signal (LOT_SIZE=65, current NIFTY lot size,
confirmed 2026-07-15). Rupee P&L per trade = spread_pnl_pts * LOT_SIZE. Daily
return = that day's total rupee P&L / Rs 50,000. This replaces an earlier
"return on premium risked" basis (COST-only denominator) that was rejected --
it ignored any shared capital constraint across trades and produced an
unusable Sharpe of ~13.5.

EMA Regime Crossover: source trades are the 2026-07-17 backtest
(ema_regime_crossover_trades.csv, pnl_rupees, 18 names). The backtest itself
was run on TOTAL_CAPITAL = Rs 1 Cr (matching OpenAlgo Sandbox convention,
12.5%-of-capital flat sizing per trade -- see ema_regime_crossover_backtest.py).
Daily return = that day's total pnl_rupees / TOTAL_CAPITAL.
"""

import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TRADING_DAYS_PER_YEAR = 252


def annualised_sharpe(daily_returns: pd.Series) -> float:
    if daily_returns.std(ddof=1) == 0:
        return float("nan")
    return daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def report(name, daily_returns: pd.Series, n_trades, n_days, basis_note):
    sharpe = annualised_sharpe(daily_returns)
    print("=" * 100)
    print(f"{name}")
    print("=" * 100)
    print(f"  Trades: {n_trades}  |  Trading days with activity: {n_days}")
    print(f"  Basis: {basis_note}")
    print(f"  Mean daily return: {daily_returns.mean():+.4%}   Std daily return: {daily_returns.std(ddof=1):.4%}")
    print(f"  Annualised Sharpe: {sharpe:.2f}   (goal: >= 2.00, gap: {2.0 - sharpe:+.2f})")
    print()
    return sharpe


# ---------------------------------------------------------------------------
# ORB_Spread — 792-trade naked-proxy backtest, spread payoff applied
# ---------------------------------------------------------------------------
WIDTH = 50
COST_PCT = 0.15
COST = WIDTH * COST_PCT  # 7.5 pts
LOT_SIZE = 65  # current NIFTY lot size, confirmed 2026-07-15
ORB_CAPITAL = 50_000  # Karan-confirmed real account size, 2026-07-18

orb = pd.read_csv(os.path.join(BASE, "orb_v2", "orb_v2_trades.csv"))
orb["exit_time"] = pd.to_datetime(orb["exit_time"])
orb["spread_pnl_pts"] = orb["pnl_pts"].clip(lower=0, upper=WIDTH) - COST
orb["spread_pnl_rupees"] = orb["spread_pnl_pts"] * LOT_SIZE
orb["exit_date"] = orb["exit_time"].dt.date

orb_daily_pnl = orb.groupby("exit_date")["spread_pnl_rupees"].sum()
orb_daily = orb_daily_pnl / ORB_CAPITAL

orb_sharpe = report(
    "ORB_Spread (50pt/15%-cost debit spread, 1 lot/signal, Rs 50,000 account)",
    orb_daily,
    n_trades=len(orb),
    n_days=len(orb_daily),
    basis_note=f"daily portfolio return = day's total spread P&L (Rs, {LOT_SIZE} qty/lot) / Rs {ORB_CAPITAL:,}",
)

# ---------------------------------------------------------------------------
# EMA Regime Crossover — 2026-07-17 backtest, Rs 1 Cr capital base
# ---------------------------------------------------------------------------
TOTAL_CAPITAL = 10_000_000  # matches ema_regime_crossover_backtest.py

ema = pd.read_csv(os.path.join(BASE, "ema_regime_crossover", "ema_regime_crossover_trades.csv"))
ema["exit_time"] = pd.to_datetime(ema["exit_time"])
ema["exit_date"] = ema["exit_time"].dt.date

ema_daily_pnl = ema.groupby("exit_date")["pnl_rupees"].sum()
ema_daily_return = ema_daily_pnl / TOTAL_CAPITAL

ema_sharpe = report(
    "EMA Regime Crossover (18-name backtest, Rs 1 Cr capital base, 12.5%/trade sizing)",
    ema_daily_return,
    n_trades=len(ema),
    n_days=len(ema_daily_return),
    basis_note="daily portfolio return = day's total pnl_rupees / Rs 1,00,00,000",
)

print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"  ORB_Spread Sharpe:            {orb_sharpe:.2f}  (goal >= 2.00)")
print(f"  EMA Regime Crossover Sharpe:  {ema_sharpe:.2f}  (goal >= 2.00)")
print("  Caveat: both computed on backtest data only (no live/forward-test equity curve exists")
print("  yet for either strategy — see indices-system/scorecard.md / equities-system spec for")
print("  current forward-test trade counts). This is a backtest-era baseline, not a live result.")
