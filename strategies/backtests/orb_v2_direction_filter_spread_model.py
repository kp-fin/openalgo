"""
ORB_Spread — Direction-Aware Trend Filter, Tested Against the ADOPTED Spread Model

Follow-up to orb_v2_backtest_direction_filter.py, which compared the current
live filter (block all entries when prev-day |net move| > 0.42%) against a
candidate direction-aware version (keep blocking big-trend LONG, but let
big-trend SHORT through) -- but only on naked spot points. Karan asked
whether that candidate's apparent improvement survives the strategy's
actual adopted spread-model economics (50pt width / 15% cost = 7.5pt cost,
intrinsic-only payoff -- see orb_v2_backtest_spread_narrow.py, the config
this vault actually adopted, NOT the 100pt/35% config in
orb_v2_backtest_spread.py which was rejected) rather than raw spot points.

Reuses the exact trade-level CSVs orb_v2_backtest_direction_filter.py
already saved (orb_v2_direction_filter/{all,current_rule,candidate_rule}_trades.csv)
-- no re-fetch, same 792-trade backtest, just re-deriving P&L per trade
under the adopted spread payoff instead of raw points.
"""

import os

import pandas as pd

WIDTH = 50
COST_PCT = 0.15
COST = WIDTH * COST_PCT  # 7.5pts

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orb_v2_direction_filter")


def load(name):
    df = pd.read_csv(os.path.join(BASE, name))
    df["spread_pnl"] = df["pnl_pts"].clip(lower=0, upper=WIDTH) - COST
    return df


def summarise(df, pnl_col, label):
    n = len(df)
    if n == 0:
        print(f"\n{label}: no trades")
        return {}
    wr = (df[pnl_col] > 0).mean() * 100
    avg = df[pnl_col].mean()
    total = df[pnl_col].sum()
    gw = df.loc[df[pnl_col] > 0, pnl_col].sum()
    gl = abs(df.loc[df[pnl_col] <= 0, pnl_col].sum())
    pf = gw / gl if gl > 0 else float("inf")
    print(f"\n-- {label} --")
    print(f"  Trades: {n} | WR: {wr:.1f}% | Avg: {avg:+.2f}pts | Total: {total:+.1f}pts | PF: {pf:.2f}")
    return {"n": n, "wr": wr, "avg": avg, "total": total, "pf": pf}


print(f"Adopted spread model: width={WIDTH}pts, cost={COST:.1f}pts (15% of width)\n")

all_trades = load("all_trades.csv")
current_rule = load("current_rule_trades.csv")   # live: quiet days only
candidate_rule = load("candidate_rule_trades.csv")  # candidate: quiet + big-trend SHORT

print("=" * 78)
print("NAKED SPOT POINTS (reference, matches the prior direction-filter test)")
print("=" * 78)
summarise(all_trades, "pnl_pts", "Unfiltered")
summarise(current_rule, "pnl_pts", "Current rule (quiet only) -- LIVE")
summarise(candidate_rule, "pnl_pts", "Candidate rule (quiet + big-trend SHORT)")

print("\n" + "=" * 78)
print(f"ADOPTED SPREAD MODEL (width={WIDTH}, cost={COST:.1f}, intrinsic-only)")
print("=" * 78)
r_all = summarise(all_trades, "spread_pnl", "Unfiltered")
r_cur = summarise(current_rule, "spread_pnl", "Current rule (quiet only) -- LIVE")
r_cand = summarise(candidate_rule, "spread_pnl", "Candidate rule (quiet + big-trend SHORT)")

print("\n" + "=" * 78)
print("BY DIRECTION -- candidate rule only (isolates what the added big-trend SHORT trades do)")
print("=" * 78)
cand_long = candidate_rule[candidate_rule["direction"] == "LONG"]
cand_short = candidate_rule[candidate_rule["direction"] == "SHORT"]
summarise(cand_long, "spread_pnl", "Candidate rule -- LONG (Bull Call Spread)")
summarise(cand_short, "spread_pnl", "Candidate rule -- SHORT (Bear Put Spread)")

# isolate exactly the added big-trend SHORT trades (in candidate, not in current)
added = candidate_rule.merge(
    current_rule[["day", "entry_time", "direction"]],
    on=["day", "entry_time", "direction"], how="left", indicator=True
)
added = added[added["_merge"] == "left_only"]
summarise(added, "spread_pnl", "Added big-trend SHORT trades only (component)")

print("\n" + "=" * 78)
print("SUMMARY -- adopted spread model")
print("=" * 78)
print(f"  {'Rule':<40}{'N':>5}{'WR%':>8}{'Avg':>9}{'Total':>10}{'PF':>7}")
for label, r in [("Unfiltered", r_all), ("Current rule (quiet only) -- LIVE", r_cur),
                  ("Candidate rule (quiet + big-trend SHORT)", r_cand)]:
    if r:
        print(f"  {label:<40}{r['n']:>5}{r['wr']:>7.1f}%{r['avg']:>9.2f}{r['total']:>10.1f}{r['pf']:>7.2f}")
