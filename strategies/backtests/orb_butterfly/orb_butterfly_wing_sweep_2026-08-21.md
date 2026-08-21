# ORB Butterfly — Wing Geometry Sweep (2026-08-21)

**Question:** among 1-2-1 butterflies (fixed lots 1/2/1), does any body/far wing pair beat ORB’s current 50pt/15% debit spread on the same post-pivot paths?

**Method:** same overlay as `orb_butterfly_overlay.py` — post-pivot LH/HL + OR_MAX≤80 (n=276). Intrinsic-only payoff; cost = 20/30/40% of each geometry’s **narrower** wing. Script: `orb_butterfly_wing_sweep.py`. Grid CSV: `orb_butterfly_wing_sweep.csv`.

**Baseline debit spread:** WR 54.0%, total +4,414 pts, **PF 5.67**.

## Grid (15 geometries × 3 cost tiers)

Includes screenshot asymmetric **150/100** (body 150, far 250), equal wings (50/50 … 200/200), and other asym pairs.

### Best per cost tier (by PF)

| Cost tier | Best wings | Cost pts | PF | vs debit 5.67 |
|---|---|---:|---:|---:|
| 20% of narrow | asym **150/50** (body 150, far 200) | 10 | **4.28** | −1.39 |
| 30% of narrow | asym **150/50** | 15 | **2.45** | −3.22 |
| 40% of narrow | asym **150/50** | 20 | **1.54** | −4.13 |

Screenshot **150/100** (body 150, far 250): PF 1.54 / 0.65 / 0.22 at 20/30/40% — mid-pack, never competitive.

Equal **50/50** is the best *equal* wing (PF 3.60 / 2.00 / 1.21) — still below debit at every tier.

**No config beats debit PF at any cost tier.**

## Interpretation

- Ranking is dominated by **assumed debit** (narrower wing → cheaper cost), not by better alignment with ORB’s +40/−25 exits. Almost no trade reaches the short body (≥50–200 pts), so intrinsic at exit ≈ `max(Δ,0)` for all geometries on this path set — wings barely differentiate until cost does.
- Tight far wing (50pt) “wins” the butterfly table only because cost scales off that 50pt wing; it still loses to the debit spread’s 7.5pt capped loss.
- Wider bodies (150–200) with 100+ far wings are worse once debit ≥20–30 pts.

## Verdict

**Rejected across the wing grid.** Keep ORB_Spread on 50pt ATM/OTM1 debit. Do not deploy any of these butterflies to Sandbox without a new reason that also redesigns exits for the body (not ORB’s +40/−25).

Caveat: intrinsic-only, no chain/IV; cost model favours narrow-wing geometries. Direction of finding (debit still wins) is robust.
