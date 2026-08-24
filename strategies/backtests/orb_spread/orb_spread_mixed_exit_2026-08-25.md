# ORB_Spread — mixed / asymmetric exits (2026-08-25)

**Question:** does the Asym gap-book mixed-exit (CE hold to Nifty weekly expiry, PE flatten 15:15) help **ORB_Spread**?

**Answer: NO.** Do not copy mixed-exit onto Spread. Host unchanged.

## Why this is a different book

Asym screenshot fly mixed-exit worked because **PE held to expiry leaked** (WR 31.7%) while CE expiry was the edge. Spread is a **50pt debit vertical**, MIS, spot +40/−25, hard 15:15.

On Spread, **PE is the stronger leg**, not the leak. And almost nobody is still open at 15:15, so an expiry policy has almost nothing to act on.

## Method

- Cache: `OpenAlgo/strategies/backtests/orb_asym/nifty_5m_cache.pkl` (2021-07-01 → 2026-06-30).
- Entries: LH/HL only, OR 30–80, range-day skip at 10:15, first signal each side, window to 12:00. **n=389** (CE 198 / PE 191).
- Exits: spot TARGET +40 / STOP −25 on 5m close. Same-day hard 15:15 vs hold to weekly expiry (Thu before 2025-09-01, Tue after; holiday roll). Mix = CE expiry path + PE same-day.
- **Primary model:** adopted debit-spread clip `clip(spot_pts, 0, 50) − 7.5`. Same convention as prior Spread overlays.
- **Overlay:** BS Δnet on ATM vs OTM1 50pt vertical, σ=15%, T to weekly expiry (theta-aware). This is the fairer test of “hold overnight”.
- ₹ ≈ pts × **65**. Sharpe = daily summed pts, mean/std × √252 (trade days only). H1/H2 split 2024-01-01.
- Live-ish extra slice: prev-day |net| ≤ 0.42% both sides, else PE only (matches live CE block). Not the primary table.

n=389 is a fresh cache regen. Older post-pivot overlay on `orb_v2_trades.csv` is n=276 (14:30 hard). Hard-exit tail is small in both (9/389 at 15:15; 21/276 at 14:30). Verdict does not depend on which n you pick.

## Clip model (adopted; no theta)

| Book | n | WR | Avg pts | ~₹/trade | Sharpe | PF | H1 WR/Sh | H2 WR/Sh |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **Baseline same-day 15:15** | 389 | 48.8% | **+14.44** | ₹938 | 14.50 | 4.78 | 48.0 / 14.38 | 50.4 / 14.83 |
| Mix CE-expiry + PE 15:15 | 389 | 49.1% | +14.78 | ₹958 | 14.92 | 4.87 | 48.4 / 14.88 | 50.4 / 15.10 |
| Both hold expiry | 389 | 49.1% | +14.84 | ₹962 | 14.97 | 4.89 | 48.4 / 14.97 | 50.4 / 15.10 |
| Control PE-expiry + CE 15:15 | 389 | 48.8% | +14.49 | ₹942 | 14.55 | 4.80 | 48.0 / 14.46 | 50.4 / 14.83 |
| Baseline **CE** | 198 | 47.0% | +13.26 | ₹862 | 9.37 | 4.37 | 45.5 / 8.82 | 50.0 / 10.40 |
| Baseline **PE** | 191 | **50.8%** | **+15.66** | ₹1,018 | 10.78 | 5.24 | 50.8 / 10.71 | 50.7 / 10.85 |

Clip Sharpe is the usual optimistic artefact (bounded −7.5 / +42.5). Use it for **ranking**, not as a live Sharpe.

## BS Δnet overlay (theta-aware) — this is the one that matters for hold

| Book | n | WR | Avg pts | ~₹/trade | Sharpe | PF | H1 Sh | H2 Sh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baseline same-day** | 389 | 49.1% | **+0.84** | ₹55 | **2.50** | 1.37 | 2.77 | 2.01 |
| Mix CE-expiry + PE 15:15 | 389 | 49.1% | +0.86 | ₹56 | 2.53 | 1.38 | 2.78 | 2.08 |
| Both hold expiry | 389 | 49.1% | +0.87 | ₹57 | 2.55 | 1.38 | 2.80 | 2.08 |
| Control PE-expiry + CE 15:15 | 389 | 49.1% | +0.84 | ₹55 | 2.51 | 1.38 | 2.80 | 2.01 |
| Baseline **CE** | 198 | 47.5% | +0.46 | ₹30 | **1.06** | 1.19 | 1.00 | 1.18 |
| Baseline **PE** | 191 | **50.8%** | **+1.23** | ₹80 | **2.86** | 1.59 | 3.34 | 2.04 |
| Mix CE-only (held) | 198 | 47.5% | +0.52 | ₹34 | 1.15 | 1.21 | 1.07 | 1.30 |

Gates for “yes it helps”: WR > 50% or at least not collapsing vs baseline, Sharpe > 2, avg pts better, n large, both halves.

Mix vs baseline: WR unchanged (49.1%), avg **+0.02 pts**, Sharpe 2.50 → 2.53. That is noise, not an edge. Combined book already clears Sharpe > 2; CE alone does **not** (1.06), and holding CE to expiry does not get it over 2.

## CE vs PE — opposite of Asym

| Leg | Baseline WR | Baseline avg (BS) | Baseline Sh (BS) | Weak? |
|---|---:|---:|---:|---|
| CE (Higher Low) | 47.5% | +0.46 | 1.06 | **Yes — under Sharpe 2** |
| PE (Lower High) | 50.8% | +1.23 | 2.86 | No |

Asym’s leak was **PE held to expiry**. Spread’s weak leg is **CE**, and it is weak on **same-day** economics, not overnight PE bleed. Holding CE to Tuesday/Thursday does not repair a same-day CE underperformance.

## Why mixed-exit cannot move this book

Exit reasons, baseline: TARGET 185 · STOP 195 · HARD_1515 **9**.

Both-expiry: TARGET 190 · STOP 197 · EXPIRY **2**.

Mix: TARGET 189 · STOP 197 · HARD_1515 2 · EXPIRY 1.

Spot +40/−25 already closes **~98%** of trades before 15:15. Expiry policy only touches the leftover handful. Asym mixed-exit worked because most flies were still open at the hard cut.

## Live-ish slice (prev-day filter)

Same ranking. Baseline CLIP n=309 WR 49.5% avg +14.84. Mix +0.33 pts. BS baseline avg +0.97 Sh 2.68 vs mix +0.99 Sh 2.69. Still no.

## Caveats

- Clip model ignores premium path; BS overlay is still flat 15% IV, no chain, no bid/ask. Overnight NRML vs live MIS is a product-type change not modelled beyond T in BS.
- Sharpe on clip is not a live estimate. BS combined Sharpe ~2.5 is closer to honest and already sits on the vault goal — mixed-exit is not what put it there.
- n=389 vs documented post-pivot 276: different generator (15:15 vs 14:30 CSV overlay). Direction identical.
- Forward Spread sample still 5 paper trades. This does not change the ≥20 gate.

## Files

- Script: `OpenAlgo/strategies/backtests/orb_spread_mixed_exit.py`
- Summaries: `orb_spread_mixed_exit_clip_summary.csv`, `orb_spread_mixed_exit_bs_summary.csv`, live-ish siblings
- Trades: `orb_spread_mixed_exit_baseline_trades.csv`, `_mix_trades.csv`, `_expiry_trades.csv`

**Next action:** leave Spread exits as they are (spot +40/−25, hard 15:15, both legs). If anything is worth a later test, it is **CE-leg quality** (already an open scorecard item after ≥20 CE forwards), not mixed-exit. Do not re-run mixed-exit on Spread without a new reason — e.g. dropping the spot target so a real overnight tail exists.
