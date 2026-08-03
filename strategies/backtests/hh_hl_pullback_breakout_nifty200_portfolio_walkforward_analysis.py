"""
HH-HL Pullback Breakout -- Nifty 200, Portfolio MC=6, Walk-Forward Split
(2026-08-03, Karan-requested follow-on to the volume-dry-up-filter result)

CONTEXT: after adding the volume dry-up filter (2026-08-03,
hh_hl_pullback_breakout.md's "Volume Dry-Up Filter Added" section), the
Nifty 200 portfolio-constrained (MAX_CONCURRENT=6) run showed Sharpe(net)
9.93 on 28 trades -- a sharp jump from the pre-filter 4.19 on 57 trades.
That magnitude of improvement on a much smaller sample, achieved by tuning
on the SAME 5-year window every other change in this campaign was also
tuned/tested on, is a classic overfitting signature. This script does not
re-run the backtest or refit anything -- it takes the ALREADY-GENERATED
trade log from hh_hl_pullback_breakout_nifty200_portfolio_backtest.py
(post-filter) and splits it chronologically into two halves by entry date,
reporting WR/PF/Sharpe/drawdown separately for each half. If the edge is
real and not a whole-period artifact, both halves should show a broadly
similar (though noisier, given smaller n) pattern. If the edge collapses in
one half, that's evidence the aggregate figure is not to be trusted.

SPLIT POINT: 2023-12-31, roughly the midpoint of the 2021-07-01->2026-06-30
backtest window (chosen by calendar midpoint, not by where the trades
happen to split most favourably -- picking the split post-hoc to make one
half look better would defeat the entire point of this exercise).

CAVEAT UP FRONT: this is NOT a true out-of-sample test in the strict sense
(the dry-up filter's threshold, 0.8x, was decided by convention/domain
knowledge, not fit to either half specifically) -- it is a stability check:
does performance hold up across time, or is it concentrated in one narrow
period. A pass here does not "confirm" the strategy; a fail is a clear red
flag. Either way this remains `inconclusive` per the vault's evidence gates.
"""

import os

import numpy as np
import pandas as pd

TRADES_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "hh_hl_pullback_breakout_nifty200_portfolio",
                           "hh_hl_pullback_breakout_nifty200_portfolio_trades.csv")
ALLOCATED_CAPITAL = 250_000  # matches the source backtest's portfolio capital
SPLIT_DATE = "2023-12-31"    # calendar midpoint of 2021-07-01 -> 2026-06-30, chosen before looking at results


def sharpe(daily_returns):
    if len(daily_returns) < 2 or daily_returns.std(ddof=1) == 0:
        return float("nan")
    return daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252)


def report(label, df):
    n = len(df)
    if n == 0:
        print(f"\n=== {label}: 0 trades ===")
        return
    wr = (df["pnl_rupees_net"] > 0).mean() * 100
    gross = df["pnl_rupees_gross"].sum()
    charges = df["total_charges"].sum()
    net = df["pnl_rupees_net"].sum()
    nw = df[df["pnl_rupees_net"] > 0]["pnl_rupees_net"].sum()
    nl = abs(df[df["pnl_rupees_net"] <= 0]["pnl_rupees_net"].sum())
    pf_net = nw / nl if nl > 0 else float("inf")

    d = df.sort_values("final_exit_time").copy()
    d["exit_date"] = pd.to_datetime(d["final_exit_time"]).dt.date
    daily_net = d.groupby("exit_date")["pnl_rupees_net"].sum() / ALLOCATED_CAPITAL
    sh = sharpe(daily_net)

    d["cum_pnl_net"] = d["pnl_rupees_net"].cumsum()
    d["peak_net"] = d["cum_pnl_net"].cummax()
    d["dd_net"] = d["cum_pnl_net"] - d["peak_net"]
    max_dd = d["dd_net"].min()

    exit_counts = df["reason"].value_counts().to_dict()

    print(f"\n=== {label}: {n} trades ===")
    print(f"Date range (entry): {df['entry_time'].min()} -> {df['entry_time'].max()}")
    print(f"WR (net): {wr:.1f}% | PF (net): {pf_net:.2f} | Net P&L: Rs {net:+,.0f} "
          f"(gross Rs {gross:+,.0f}, charges Rs {charges:,.0f})")
    print(f"Sharpe (net): {sh:.2f} | Max DD (net): Rs {max_dd:,.0f} ({abs(max_dd)/ALLOCATED_CAPITAL*100:.1f}% of capital)")
    print(f"Exit breakdown: {exit_counts}")
    if "STOP_BEFORE_LEG1" in exit_counts or "TRAIL_STOP" in exit_counts:
        for reason, g in df.groupby("reason"):
            print(f"  {reason:20s} n={len(g):3d}  WR={(g.pnl_rupees_net>0).mean()*100:5.1f}%  "
                  f"avg_pnl_pct={g.pnl_pct_blended.mean()*100:+.2f}%  avg_hold={g.hold_days.mean():.1f}d")


trades_df = pd.read_csv(TRADES_CSV, parse_dates=["entry_time", "final_exit_time"])
trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)

split_ts = pd.Timestamp(SPLIT_DATE, tz=trades_df["entry_time"].dt.tz)
first_half = trades_df[trades_df["entry_time"] < split_ts]
second_half = trades_df[trades_df["entry_time"] >= split_ts]

print(f"Full trade log: {len(trades_df)} trades, split at {SPLIT_DATE} (calendar midpoint, fixed before viewing results)")

report("FULL PERIOD (2021-07-01 -> 2026-06-30)", trades_df)
report(f"FIRST HALF (entries before {SPLIT_DATE})", first_half)
report(f"SECOND HALF (entries on/after {SPLIT_DATE})", second_half)

print("\n--- Stability read ---")
print("If WR/PF/Sharpe are broadly similar (same sign, same order of magnitude,")
print("not one half carrying the entire edge) across both halves, that's evidence")
print("the aggregate figure isn't a one-period artifact. If one half is flat or")
print("negative while the other explains ~all the P&L, treat the full-period")
print("Sharpe as unreliable -- concentrated in a narrow window, not a durable edge.")
