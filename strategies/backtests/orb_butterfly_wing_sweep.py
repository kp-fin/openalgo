"""
ORB butterfly wing-geometry sweep (2026-08-21)

Research-only. Does not change live ORB_Spread.

Same spot-path overlay as orb_butterfly_overlay.py, but sweeps body/far
strike distances (NIFTY 50pt multiples) for the fixed 1-2-1 butterfly:
  BUY ATM x1, SELL ATM+/-body x2, BUY ATM+/-far x1

Includes the screenshot asymmetric (150/250) and equal-wing neighbours.
Cost = {20%, 30%, 40%} of the narrower wing for each geometry.
Primary slice: post-pivot LH/HL + OR_MAX<=80.
"""

import os

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TRADES_CSV = os.path.join(BASE, "orb_v2", "orb_v2_trades.csv")
OUT_DIR = os.path.join(BASE, "orb_butterfly")
os.makedirs(OUT_DIR, exist_ok=True)

DEBIT_SPREAD_WIDTH = 50
DEBIT_SPREAD_COST = DEBIT_SPREAD_WIDTH * 0.15
COST_PCTS = (0.20, 0.30, 0.40)

# (body_pts, far_pts) — far must be > body; both multiples of 50.
# tag notes intent for the write-up.
WINGS = [
    (50, 100, "equal 50/50"),
    (100, 200, "equal 100/100"),
    (150, 300, "equal 150/150"),
    (200, 400, "equal 200/200"),
    (50, 150, "asym 50/100"),
    (50, 200, "asym 50/150"),
    (100, 150, "asym 100/50"),
    (100, 250, "asym 100/150"),
    (100, 300, "asym 100/200"),
    (150, 200, "asym 150/50"),
    (150, 250, "asym 150/100 screenshot"),
    (150, 350, "asym 150/200"),
    (200, 300, "asym 200/100"),
    (200, 350, "asym 200/150"),
    (200, 450, "asym 200/250"),
]


def butterfly_intrinsic(delta, body, far):
    return (
        max(delta - 0.0, 0.0)
        - 2.0 * max(delta - body, 0.0)
        + max(delta - far, 0.0)
    )


def metrics(series):
    n = len(series)
    if n == 0:
        return {"n": 0, "wr": float("nan"), "avg": float("nan"),
                "total": 0.0, "pf": float("nan")}
    gw = series[series > 0].sum()
    gl = abs(series[series <= 0].sum())
    return {
        "n": n,
        "wr": (series > 0).mean() * 100,
        "avg": series.mean(),
        "total": series.sum(),
        "pf": gw / gl if gl > 0 else float("inf"),
    }


trades = pd.read_csv(TRADES_CSV)
trades["or_width"] = trades["orb_high"] - trades["orb_low"]
post = trades[
    trades["signal"].isin(["HigherLow", "LowerHigh"]) & (trades["or_width"] <= 80)
].copy()

debit = post["pnl_pts"].clip(lower=0, upper=DEBIT_SPREAD_WIDTH) - DEBIT_SPREAD_COST
base = metrics(debit)

print(f"Post-pivot LH/HL + OR<=80  n={len(post)}")
print(f"Baseline debit 50pt/15% (cost {DEBIT_SPREAD_COST}): "
      f"WR={base['wr']:.1f}%  avg={base['avg']:+.2f}  total={base['total']:+.1f}  PF={base['pf']:.2f}")
print()

rows = []
for body, far, tag in WINGS:
    narrow = min(body, far - body)
    wide = max(body, far - body)
    equal = body == (far - body)
    for pct in COST_PCTS:
        cost = narrow * pct
        pnl = post["pnl_pts"].map(lambda d, b=body, f=far: butterfly_intrinsic(d, b, f)) - cost
        m = metrics(pnl)
        rows.append({
            "body": body,
            "far": far,
            "wing_near": body,
            "wing_far": far - body,
            "narrow_wing": narrow,
            "wide_wing": wide,
            "equal_wings": equal,
            "tag": tag,
            "cost_pct": pct,
            "cost_pts": cost,
            **m,
            "vs_debit_pf": m["pf"] - base["pf"],
            "beats_debit": m["pf"] > base["pf"],
        })

df = pd.DataFrame(rows)
out_csv = os.path.join(OUT_DIR, "orb_butterfly_wing_sweep.csv")
df.to_csv(out_csv, index=False)

# Rank within each cost tier by PF
print("=" * 100)
print("WING SWEEP — ranked by PF within each cost tier (post-pivot)")
print("=" * 100)
for pct in COST_PCTS:
    sub = df[df["cost_pct"] == pct].sort_values("pf", ascending=False)
    print(f"\n--- Cost = {pct*100:.0f}% of narrower wing ---")
    print(f"  {'body':>5} {'far':>5} {'wings':>10} {'cost':>6} {'WR':>6} {'avg':>8} {'total':>10} {'PF':>7}  tag")
    for _, r in sub.iterrows():
        wings = f"{int(r['wing_near'])}/{int(r['wing_far'])}"
        mark = " <<" if r["beats_debit"] else ""
        print(f"  {int(r['body']):5d} {int(r['far']):5d} {wings:>10} {r['cost_pts']:6.0f} "
              f"{r['wr']:5.1f}% {r['avg']:+7.2f} {r['total']:+9.1f} {r['pf']:7.2f}  {r['tag']}{mark}")
    best = sub.iloc[0]
    print(f"  Best: body={int(best['body'])} far={int(best['far'])} PF={best['pf']:.2f} "
          f"vs debit PF={base['pf']:.2f} (delta {best['vs_debit_pf']:+.2f})")

# Mid-tier (30%) summary: anything remotely competitive?
mid = df[df["cost_pct"] == 0.30].sort_values("pf", ascending=False)
print("\n" + "=" * 100)
print("MID COST (30% of narrow wing) — top 5 vs debit baseline")
print("=" * 100)
print(mid.head(5)[["body", "far", "tag", "cost_pts", "wr", "avg", "total", "pf", "vs_debit_pf"]]
      .to_string(index=False))
print(f"\nAny config beats debit PF {base['pf']:.2f} at 30% cost? "
      f"{bool(mid['beats_debit'].any())}")
print(f"Any config beats debit at ANY cost tier? {bool(df['beats_debit'].any())}")

# Optimistic 20% tier — still useful to see if geometry alone can close the gap
opt = df[df["cost_pct"] == 0.20].sort_values("pf", ascending=False)
print(f"\nOptimistic 20% tier best PF={opt.iloc[0]['pf']:.2f} "
      f"({opt.iloc[0]['tag']}) vs debit {base['pf']:.2f}")

print(f"\nFull grid -> {out_csv}")
print("Caveat: intrinsic-only; cost scaled to each geometry's narrower wing. "
      "Directional evidence only.")
