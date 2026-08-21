# ORB + 1-2-1 Asymmetric Butterfly Overlay (2026-08-21)

**Question:** does replacing ORB_Spread’s 50pt debit spread with a fixed 1-2-1 butterfly (ATM / OTM3 / OTM5, lots 1/2/1) improve performance?

**Method:** overlay on existing `orb_v2_trades.csv` spot paths (same entries/exits). Intrinsic-only butterfly payoff minus assumed net debit; compare to adopted debit spread (50pt width, 15% cost = 7.5pt). Script: `orb_butterfly_overlay.py`.

**Structure:**
- Bull: BUY ATM CE ×1, SELL ATM+150 CE ×2, BUY ATM+250 CE ×1
- Bear: BUY ATM PE ×1, SELL ATM−150 PE ×2, BUY ATM−250 PE ×1
- Fixed quantity (no capital sizing)

**Cost sweep:** debit = 20% / 30% / 40% of narrower wing (100pt) → 20 / 30 / 40 pts.

## Primary result — post-pivot (LH/HL + OR_MAX≤80, n=276)

| Vehicle | WR | Avg pts | Total pts | PF |
|---|---:|---:|---:|---:|
| Debit spread 50pt/15% (**current**) | 54.0% | +16.0 | +4,414 | **5.67** |
| Butterfly debit 20pt (optimistic) | 51.4% | +5.0 | +1,390 | 1.54 |
| Butterfly debit 30pt | 51.4% | −5.0 | −1,370 | 0.65 |
| Butterfly debit 40pt | 49.6% | −15.0 | −4,130 | 0.22 |

Closest-to-live slice (post-pivot ∩ direction-aware candidate rule, n=224): same pattern — debit PF **6.12** vs butterfly best-case PF **1.69** (20pt debit), then sub-1 at 30/40.

## Why it fails on ORB paths

ORB exits are spot ±40 / −25. At a +40 target the butterfly’s intrinsic is only ~40pts (still below the body at 150), so after a realistic debit it barely clears — while a stopped-out trade loses the **full debit**. The debit spread caps loss at ~7.5pts by construction. Butterfly max profit sits near Δ=+150, which this strategy almost never holds for.

## Verdict

**Rejected.** Do not replace ORB_Spread’s vehicle with this 1-2-1 butterfly. Do not deploy to Sandbox. Keep 50pt ATM/OTM1 debit spread, equal lots (capital-scaled as today).

**Not a free quantity tweak** — it is a different payoff shape that fights ORB’s own target/stop geometry.

Caveat: intrinsic-only, no chain/IV. Direction of finding (butterfly much worse) is robust across the debit sweep; only the optimistic 20pt tier stays net-positive, and even then far behind the debit spread.

Files: `orb_butterfly_overlay_summary.csv`, `orb_butterfly_postpivot_trades.csv`.
