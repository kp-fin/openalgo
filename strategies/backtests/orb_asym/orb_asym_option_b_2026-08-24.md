# ORB_Asym option B backtest (2026-08-24)

**B =** skip if live/quoted fly net ≤ 0, plus flatten if an open leg fails.  
Flatten is execution-only — **not** in this grid.

**Method:** locked v2 255 paths; skip when BS entry net ≤ 0. Same files as `orb_asym_live_mtm_2026-08-24.md`. CSV: `orb_asym_option_b_sweep.csv`.

Intrinsic v2 **cannot** skip anything (every trade is booked at +7.5 debit).

## Results (MTM pts)

| σ | Rule | n | skipped | WR | PF | Sharpe | Net pts | Note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| — | Locked intrinsic, no skip | 255 | 0 | 50.6 | 5.71 | 12.77 | +4,401 | Not live P&L |
| 10% | B off | 255 | 0 | 79.2 | 3.98 | 7.48 | +214 | |
| 10% | **B on** | **56** | 199 | 94.6 | 17.87 | 18.38 | +83 | **53/56 are DTE=0** |
| 12% | B on | 34 | 221 | 97.1 | 288 | 26.2 | +58 | **34/34 DTE=0** |
| 15% | B on | 9 | 246 | 100 | inf | 40.6 | +15 | toy sample |
| 20% | B on | **0** | 255 | — | — | — | 0 | blank book |

H2 at 10% B on: **n=18**. At 12%: **n=3**.

## What this means

In BS, a 1-2-1 is a **credit** whenever there is time value on the short body. A **debit** print is almost only **expiry day** (intrinsic). So option B does **not** “keep the good v2 trades and drop 24 Aug.” It **throws away ~80–100% of days** and **keeps expiry**.

24 Aug (DTE=1, net −0.50) **would skip** — that part works. The cost is a strategy that barely trades, clustered on **weekly expiry**, which we already know is the ugly regime for this fly.

**Verdict: do not adopt B as a backtest winner.** Flatten-on-fail is still a live safety patch (separate, no stats). Credit skip remains an operational bandage that this model says will starve the book.

**Not frozen into v2.**
