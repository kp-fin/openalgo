# Gap-and-Go — LONG gap-band A/B (2026-08-19)

**Question:** does loosening the deployed LONG 3–10% filter restore activity without killing the edge?

**Method:** one fetch of the deployed 60-name universe (2021-07-01 → 2026-06-30, vol lookback 20d), then re-run the same portfolio sim (6 concurrent, 2% daily halt, charges) for each LONG band. SHORTs keep `|gap| >= 1.5%` with no upper cap; only the LONG band changes. Script: `gap_and_go_backtest.py` with `GAG_AB=1`.

**Caveats:**
- CGPOWER + ADANIENSOL skipped (known Dhan interior-chunk gap) — same as prior runs.
- TMCV skipped this run (DH-904 rate limit near end of fetch). Candidate count 1,008 vs prior ~1,033-ish signal pool; absolute levels shift slightly vs the Aug-6 published table, but **band ranking within this run is the fair comparison**.
- Concurrency means accepted SHORT sets are not byte-identical across bands (same SHORT *n* here, different which LONGs occupied slots). Treat SHORT totals as mildly contaminated; LONG-band ranking is still the point.

## Results

| Band | Trades | LONG / SHORT | WR (net) | PF (net) | Sharpe (net) | Net P&L | Max DD (net) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unfiltered (LONG ≥1.5%, no upper) | 961 | 685 / 276 | 48.3% | 1.22 | 1.18 | +₹1,67,743 | 19.1% |
| LONG 1.5–10% | 911 | 635 / 276 | 48.8% | 1.26 | 1.38 | +₹1,89,541 | 19.3% |
| LONG 2–10% | 694 | 418 / 276 | 49.7% | 1.31 | **1.55** | +₹1,73,228 | 16.8% |
| LONG 3–10% (deployed) | 519 | 243 / 276 | **51.6%** | **1.38** | **1.77** | +₹1,53,417 | **10.8%** |

Summary CSV: `gap_and_go_long_band_ab_vol20.csv`. Per-band trade logs written under the same folder.

## Verdict

- **Keep 3–10% if quality is the goal.** Best Sharpe, PF, WR, and roughly half the max DD of the looser bands. Matches why it was deployed.
- **2–10% is the only loosening that still clears Sharpe(net) ≥ 1.5** (+175 LONG trades vs deployed: 418 vs 243). Accept ~0.22 Sharpe give-up and deeper DD for more activity.
- **1.5–10% maximises rupee P&L but fails the Sharpe gate (1.38).** Do not deploy for activity alone.
- Caps at 10% still help vs fully unfiltered (1.5–10% Sharpe 1.38 vs unfiltered 1.18) — the upper trim is real; the lower floor is what trades activity for edge.

**Not a live config change.** Paper log since Aug-6 still has ~0 trades under 3–10%; that is selectivity + breakout gate, not proof the band is wrong. Loosening is a deliberate edge-vs-activity trade-off, not a free lunch.

## Next Action

- Default: leave live filter at LONG 3–10%.
- Only if paper inactivity is unacceptable: trial LONG 2–10% in Sandbox only, with a dated log note and a pre-agreed review after N≥30 filtered-era trades (or 8 weeks).
