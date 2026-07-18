"""
ORB_Spread vs EMA Regime Crossover — Daily P&L Correlation
Backtest-based (2021-07-01 -> 2026-06-30) -- live data is only 1 day old for
EMA Regime Crossover (went live 2026-07-17), far too thin for a meaningful
live correlation. Uses each strategy's own current-config backtest trade log:

  ORB_Spread: strategies/backtests/orb_v2_dow_expiry_regime_aware/all_trades.csv
    (current live config: HARD_EXIT 15:15, standard entry logic, 792 trades)
  EMA Regime Crossover: strategies/backtests/ema_regime_crossover/ema_regime_crossover_trades.csv
    (current live config: long+short both, 12.5%/6-position cap, 10,348 trades)

Method: aggregate each strategy's trades to a daily P&L series (sum of that
day's closed trades), align on calendar date over the shared window, treat a
day with no trades for a strategy as 0 P&L for that strategy (not dropped --
a flat day is a real data point for correlation purposes), then compute the
Pearson correlation coefficient between the two daily P&L series.

Units differ (ORB_Spread in spot points, EMA in rupees) but correlation is
scale-invariant so this doesn't matter for the coefficient itself.
"""

import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

orb = pd.read_csv(os.path.join(BASE, "orb_v2_dow_expiry_regime_aware", "all_trades.csv"))
ema = pd.read_csv(os.path.join(BASE, "ema_regime_crossover", "ema_regime_crossover_trades.csv"))

orb["date"] = pd.to_datetime(orb["day"]).dt.date
ema["date"] = pd.to_datetime(ema["entry_time"]).dt.date

orb_daily = orb.groupby("date")["pnl_pts"].sum().rename("orb_spread_pnl")
ema_daily = ema.groupby("date")["pnl_rupees"].sum().rename("ema_regime_crossover_pnl")

# Full calendar-day index across the shared window, both strategies zero-filled
# on days they didn't trade (a flat day is a real, meaningful data point here).
start = min(orb_daily.index.min(), ema_daily.index.min())
end   = max(orb_daily.index.max(), ema_daily.index.max())
all_days = pd.date_range(start, end, freq="D").date

combined = pd.DataFrame(index=all_days)
combined["orb_spread_pnl"] = orb_daily.reindex(all_days).fillna(0)
combined["ema_regime_crossover_pnl"] = ema_daily.reindex(all_days).fillna(0)

# Restrict to weekdays only (both are equity/index intraday strategies -- no
# weekend trading days exist in either source, this just drops the zero-filled
# weekend rows so they don't dilute the correlation with meaningless 0-vs-0 pairs).
weekday_mask = pd.to_datetime(combined.index).dayofweek < 5
combined = combined[weekday_mask]

print(f"Shared window: {start} -> {end} ({len(combined)} weekdays)")
print(f"ORB_Spread: {(combined['orb_spread_pnl'] != 0).sum()} days with a trade, "
      f"{(combined['orb_spread_pnl'] == 0).sum()} flat days")
print(f"EMA Regime Crossover: {(combined['ema_regime_crossover_pnl'] != 0).sum()} days with a trade, "
      f"{(combined['ema_regime_crossover_pnl'] == 0).sum()} flat days")
both_active = ((combined["orb_spread_pnl"] != 0) & (combined["ema_regime_crossover_pnl"] != 0)).sum()
print(f"Days BOTH strategies traded: {both_active}")

corr_all = combined["orb_spread_pnl"].corr(combined["ema_regime_crossover_pnl"])
print(f"\nPearson correlation, all weekdays (zero-filled): {corr_all:.4f}")

both_only = combined[(combined["orb_spread_pnl"] != 0) & (combined["ema_regime_crossover_pnl"] != 0)]
if len(both_only) >= 2:
    corr_both = both_only["orb_spread_pnl"].corr(both_only["ema_regime_crossover_pnl"])
    print(f"Pearson correlation, days BOTH traded only (n={len(both_only)}): {corr_both:.4f}")
else:
    print(f"Too few days ({len(both_only)}) where both traded to compute a second-view correlation.")

combined.to_csv(os.path.join(BASE, "orb_spread_ema_daily_pnl.csv"))
print(f"\nDaily P&L series saved to orb_spread_ema_daily_pnl.csv")
