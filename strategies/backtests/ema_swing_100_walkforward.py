"""
EMA Swing-100 (High-Beta Top-15 from Nifty 100) -- Walk-Forward Split Check (2026-08-04)

Splits the already-generated 5-year trade log
(ema_regime_crossover_swing_cnc_high_beta_nifty100_trades.csv, portfolio-simulated
with the same MAX_CONCURRENT=6 / MAX_PER_GROUP=2 constraints as the full run) at the
window midpoint and recomputes WR/PF/Sharpe/maxDD independently for each half -- the
same check that caught HH-HL Pullback Breakout's inflated aggregate Sharpe (9.93
aggregate vs. a much more modest 5.40 second-half) and confirmed Gap-and-Go's aggregate
held up under a split. Trades aren't re-simulated -- the portfolio-level constraints
were already applied causally in chronological order during the original run;
splitting the accepted-trade list post-hoc for reporting is the same method used by
both prior walk-forward checks in this vault.
"""

import os
import numpy as np
import pandas as pd

IN_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ema_regime_crossover_swing_cnc_high_beta_nifty100",
    "ema_regime_crossover_swing_cnc_high_beta_nifty100_trades.csv",
)
ALLOCATED_CAPITAL = 250_000

df = pd.read_csv(IN_CSV, parse_dates=["entry_time", "exit_time"])
df = df.sort_values("entry_time").reset_index(drop=True)

start, end = df["entry_time"].min(), df["entry_time"].max()
midpoint = start + (end - start) / 2
print(f"Window: {start.date()} -> {end.date()} | Midpoint: {midpoint.date()}")

first_half = df[df["entry_time"] < midpoint]
second_half = df[df["entry_time"] >= midpoint]


def summarize(sub, label):
    if sub.empty:
        print(f"\n=== {label}: no trades ===")
        return
    n = len(sub)
    wr_net = (sub["net_pnl_rupees"] > 0).mean() * 100
    nw = sub[sub["net_pnl_rupees"] > 0]["net_pnl_rupees"].sum()
    nl = abs(sub[sub["net_pnl_rupees"] <= 0]["net_pnl_rupees"].sum())
    pf_net = nw / nl if nl > 0 else float("inf")
    total_net = sub["net_pnl_rupees"].sum()

    sub = sub.copy()
    sub["exit_date"] = sub["exit_time"].dt.date
    daily_net = sub.groupby("exit_date")["net_pnl_rupees"].sum() / ALLOCATED_CAPITAL
    if len(daily_net) >= 2 and daily_net.std(ddof=1) > 0:
        sharpe_net = daily_net.mean() / daily_net.std(ddof=1) * np.sqrt(252)
    else:
        sharpe_net = float("nan")

    dd = sub.sort_values("exit_time").reset_index(drop=True)
    dd["cum"] = dd["net_pnl_rupees"].cumsum()
    dd["peak"] = dd["cum"].cummax()
    dd["dd"] = dd["cum"] - dd["peak"]
    max_dd_pct = abs(dd["dd"].min()) / ALLOCATED_CAPITAL * 100

    print(f"\n=== {label} ({sub['entry_time'].min().date()} -> {sub['entry_time'].max().date()}) ===")
    print(f"Trades: {n}")
    print(f"WR (net): {wr_net:.1f}% | PF (net): {pf_net:.2f} | Net P&L: Rs {total_net:+,.0f}")
    print(f"Sharpe (net): {sharpe_net:.2f} | Max drawdown (net): {max_dd_pct:.1f}%")


summarize(df, "FULL WINDOW (reference)")
summarize(first_half, "FIRST HALF")
summarize(second_half, "SECOND HALF")
