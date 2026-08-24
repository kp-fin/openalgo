# PE-leg sharpen — Nifty gap+VWAP 1-2-1 (2026-08-24)

**Question:** can the down-gap **PE** fly clear WR > 50% and Sharpe > 2 without copying CE’s hold-to-Tuesday (which prints +14.6 pts on calls and **31.7% WR** on puts)?

Locked gap set: **0.58–3.5%** + 10:00 VWAP, screenshot PE 24150/24000×2/23900 analog. Nifty Tuesday expiry (Thu before 1 Sep 2025). No DTE-skip. No Defensive OR-Hold.

## Diagnosis

| PE exit | n | WR | Sharpe | Avg pts | Avg ₹ | H1 | H2 |
|---|---:|---:|---:|---:|---:|---|---|
| Hold to expiry | 41 | **31.7%** | 0.15 | +0.38 | ₹25 | 30% / 0.34 | 33% / −0.05 |
| **15:15 same day** | **40** | **55.0%** | **2.88** | **+4.78** | **₹311** | **55% / 3.15** | **55% / 2.50** |

DTE≥3 PE expiry: n=26 WR **23%** avg **−11.3**. That is the leak. Same-day those 26 still WR 54%.

Debit floor on PE expiry does not help (debit>15 WR still 32%).

## PE same-day grid (n≥30)

On `g58_35_1000`, **hold to 15:15** is the best avg that still clears both halves. Trails/MTM raise Sharpe on *tighter* gap caps (n=33) but cut avg and H2 sample.

| PE vehicle | n | WR | Sh | Avg | H2 |
|---|---:|---:|---:|---:|---|
| **hold 15:15** | 40 | 55.0 | 2.88 | **+4.78** | 55 / 2.50 |
| trail 8/5 | 40 | 60.0 | 2.34 | +1.38 | 50 / 1.57 |
| mtm 16/8 | 40 | 50.0 | 3.06 | +2.18 | 50 / 1.58 |

Not adopted: `g58_18_1000` + mtm_16_8 (n=33 Sh 3.99 avg +2.75, H2 n=17).

## Mixed book (the actual Asym)

**CE hold to Nifty weekly expiry + PE flatten 15:15**, same gap+VWAP entries.

| | n | WR | PF | Sharpe | Avg pts | Avg ₹ | H1 | H2 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Mix | 107 | **60.7%** | 2.21 | **4.33** | **+10.52** | **₹684** | 59.0 / 4.33 | 63.0 / 4.38 |
| CE expiry only | 67 | 64.2 | 2.26 | 5.02 | +13.95 | ₹907 | 61.0 / 4.79 | 69.2 / 5.46 |
| PE 15:15 only | 40 | 55.0 | 2.01 | 2.88 | +4.78 | ₹311 | 55.0 / 3.15 | 55.0 / 2.50 |

Vs both-sides hold-to-expiry: WR 51.9 → **60.7**, avg +9.10 → **+10.52**, PE no longer a 32% WR drag.

**PE will not match CE’s +14 pts** on this vehicle. Calls get the week; puts get the afternoon. That is the edge, not a tighter put debit.

## Caveats

- Mix n=107 vs 106 on the earlier expiry CSV (one extra paired bar). Same story.
- PE n=40 is thin vs CE 67. H2 PE n=20.
- CE expiry is NRML; PE is same-day — Host must use **NRML for CE** and **MIS (or same-day NRML flatten)** for PE. Two product types in one strategy.
- Not deployed. Frozen v2 Host stays.

Grid: `orb_asym_pe_sharpen.csv`. Mix: `orb_asym_pe_mix_trades.csv`.

**Next action:** paper **asymmetric exits** (CE→Tuesday, PE→15:15) if you want this book live. Do not hold PE to expiry.
