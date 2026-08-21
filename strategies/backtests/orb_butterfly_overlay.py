"""
ORB + 1-2-1 asymmetric butterfly overlay (2026-08-21)

Research-only. Does not change live ORB_Spread.

Keeps ORB spot entries/exits from orb_v2_trades.csv; replaces the options
vehicle with a fixed 1-2-1 butterfly (ATM / OTM3 / OTM5 = 0 / +150 / +250
in the trade direction) and compares to the adopted 50pt/15% debit spread
on the same trades.

Structure (per plan):
  Bull (LONG/CE): BUY ATM x1, SELL ATM+150 x2, BUY ATM+250 x1
  Bear (SHORT/PE): BUY ATM x1, SELL ATM-150 x2, BUY ATM-250 x1
  Fixed lots — no capital scaling.

Payoff: intrinsic-only change from entry (ATM ≈ entry spot), minus assumed
net debit. Same caveat as prior ORB spread overlays — no historical chain/IV.

Cost sweep: debit = {20%, 30%, 40%} of the narrower wing (100pts).
"""

import os

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TRADES_CSV = os.path.join(BASE, "orb_v2", "orb_v2_trades.csv")
CAND_CSV = os.path.join(BASE, "orb_v2_direction_filter", "candidate_rule_trades.csv")
OUT_DIR = os.path.join(BASE, "orb_butterfly")
os.makedirs(OUT_DIR, exist_ok=True)

BODY_PTS = 150   # OTM3
FAR_PTS = 250    # OTM5
NARROW_WING = FAR_PTS - BODY_PTS  # 100
DEBIT_SPREAD_WIDTH = 50
DEBIT_SPREAD_COST = DEBIT_SPREAD_WIDTH * 0.15  # 7.5
COST_PCTS = (0.20, 0.30, 0.40)


def butterfly_intrinsic(delta):
    """Intrinsic value of 1-2-1 butterfly given spot move in trade direction.

    Strikes at 0 / +BODY / +FAR relative to entry ATM. At entry (delta=0)
    all legs are at-the-money / OTM so intrinsic is ~0; exit value is the
    change in intrinsic.
    """
    return (
        max(delta - 0.0, 0.0)
        - 2.0 * max(delta - BODY_PTS, 0.0)
        + max(delta - FAR_PTS, 0.0)
    )


def summarise(series, label):
    n = len(series)
    if n == 0:
        print(f"  {label}: no trades")
        return {"label": label, "n": 0, "wr": float("nan"), "avg": float("nan"),
                "total": 0.0, "pf": float("nan")}
    wr = (series > 0).mean() * 100
    avg = series.mean()
    total = series.sum()
    gw = series[series > 0].sum()
    gl = abs(series[series <= 0].sum())
    pf = gw / gl if gl > 0 else float("inf")
    print(f"  {label:<42} n={n:4d}  WR={wr:5.1f}%  avg={avg:+7.2f}  "
          f"total={total:+9.1f}  PF={pf:6.2f}")
    return {"label": label, "n": n, "wr": wr, "avg": avg, "total": total, "pf": pf}


def run_slice(df, slice_name, rows_out):
    print(f"\n{'=' * 90}")
    print(f"{slice_name}  (n={len(df)})")
    print(f"{'=' * 90}")
    if df.empty:
        print("  empty slice")
        return

    debit_pnl = df["pnl_pts"].clip(lower=0, upper=DEBIT_SPREAD_WIDTH) - DEBIT_SPREAD_COST
    summarise(df["pnl_pts"], "Naked spot pts (reference)")
    summarise(debit_pnl, f"Debit spread 50pt/15% (cost {DEBIT_SPREAD_COST})")

    for pct in COST_PCTS:
        cost = NARROW_WING * pct
        bf_pnl = df["pnl_pts"].map(butterfly_intrinsic) - cost
        r = summarise(bf_pnl, f"Butterfly 1-2-1 debit={cost:.0f}pt ({pct*100:.0f}% of {NARROW_WING})")
        rows_out.append({"slice": slice_name, "vehicle": r["label"], **{k: r[k] for k in ("n", "wr", "avg", "total", "pf")}})

        # direction split
        for direction, g in df.groupby("direction"):
            g_pnl = g["pnl_pts"].map(butterfly_intrinsic) - cost
            summarise(g_pnl, f"  {direction} only @ debit {cost:.0f}")

    rows_out.append({
        "slice": slice_name,
        "vehicle": f"Debit spread 50pt/15% (cost {DEBIT_SPREAD_COST})",
        "n": len(df),
        "wr": (debit_pnl > 0).mean() * 100,
        "avg": debit_pnl.mean(),
        "total": debit_pnl.sum(),
        "pf": (debit_pnl[debit_pnl > 0].sum() / abs(debit_pnl[debit_pnl <= 0].sum())
               if (debit_pnl <= 0).any() else float("inf")),
    })


trades = pd.read_csv(TRADES_CSV)
trades["or_width"] = trades["orb_high"] - trades["orb_low"]

# Closest-to-live: LH/HL only + OR_MAX=80, intersected with direction-aware
# candidate rule (quiet + big-trend SHORT) adopted 2026-08-04.
cand = pd.read_csv(CAND_CSV)
cand_keys = set(cand["entry_time"].astype(str))

post = trades[
    trades["signal"].isin(["HigherLow", "LowerHigh"]) & (trades["or_width"] <= 80)
].copy()
liveish = post[post["entry_time"].astype(str).isin(cand_keys)].copy()

rows = []
run_slice(trades, "FULL baseline (792, all signals, OR<=150 era)", rows)
run_slice(post, "POST-PIVOT LH/HL + OR_MAX<=80", rows)
run_slice(liveish, "CLOSEST-TO-LIVE (post-pivot ∩ candidate prev-day rule)", rows)

summary = pd.DataFrame(rows)
summary_csv = os.path.join(OUT_DIR, "orb_butterfly_overlay_summary.csv")
summary.to_csv(summary_csv, index=False)

# Persist the primary slice with both vehicles at the mid cost tier for audit.
mid_cost = NARROW_WING * 0.30
audit = post.copy()
audit["debit_spread_pnl"] = audit["pnl_pts"].clip(lower=0, upper=DEBIT_SPREAD_WIDTH) - DEBIT_SPREAD_COST
audit["butterfly_intrinsic"] = audit["pnl_pts"].map(butterfly_intrinsic)
audit["butterfly_pnl_debit30"] = audit["butterfly_intrinsic"] - mid_cost
audit_csv = os.path.join(OUT_DIR, "orb_butterfly_postpivot_trades.csv")
audit.to_csv(audit_csv, index=False)

print(f"\nSummary -> {summary_csv}")
print(f"Post-pivot trade audit -> {audit_csv}")
print("\nCaveat: intrinsic-only butterfly proxy; real same-day exits keep time value. "
      "Treat PF as directional evidence, not precise prediction.")
