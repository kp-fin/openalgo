# ORB_Asym v2 — mixed-exit check (2026-08-24)

**Question:** does the gap-book PE fix (CE → Nifty weekly expiry, PE → 15:15) also apply to **v2** (Defensive OR-Hold, ATM / body 50 / far 150)?

Same 255 locked v2 paths. Mark = BS Δnet σ=15%, strikes **round-to-50 ATM** (not screenshot floor50−100). Calendar: Nifty Thu until 31 Aug 2025, Tue after. ₹ = pts × 65.

## Result — do **not** copy mixed-exit onto v2

v2 entries are **credits** (CE 97.7% / PE 95.3%, mean net0 ≈ −11). Screenshot gap fly was a **debit**. Holding a credit to expiry is theta; holding a debit put to expiry was the leak.

| Book | n | WR | Sharpe | Avg pts | Avg ₹ |
|---|---:|---:|---:|---:|---:|
| v2 both sides, **spot 50/−25 BS** (live-like) | 255 | 76.5% | 6.85 | +0.87 | ₹57 |
| PE v2-spot BS | 127 | **78.0%** | 9.17 | +0.85 | ₹55 |
| PE hold **15:15** | 127 | 86.6% | 11.68 | +4.38 | ₹284 |
| PE hold **expiry** | 127 | **98.4%** | 29.40 | **+16.84** | ₹1,095 |
| CE hold expiry | 128 | 99.2% | 27.89 | +17.32 | ₹1,126 |
| Mix CE-exp + PE-15:15 | 255 | 92.9% | 19.26 | +10.88 | ₹707 |
| **Both hold expiry** | 255 | **98.8%** | 26.13 | **+17.08** | ₹1,110 |

On v2, PE is **not** the weak expiry leg. Mixed-exit **cuts** average vs both-to-expiry (+10.88 vs +17.08).

H1/H2 on PE expiry: WR 96.6 / 100. Same story both halves.

## Contrast with screenshot gap book

| | PE expiry WR | PE 15:15 WR |
|---|---:|---:|
| v2 credit 50/150 | **98.4%** | 86.6% |
| Gap+VWAP debit 150/250 | **31.7%** | **55.0%** |

Asymmetric exits are for the **debit screenshot fly**, not for frozen v2.

## Caveats

- 99% WR / +17 pts on v2 expiry is **credit collection to T=0**, not a TARGET +42 intrinsic. Live 24 Aug already showed a v2 **credit** marking against you on a spot win.
- No bid-ask. NRML overnight on a short-heavy credit is gap risk the 255-path BS file does not price.
- Do not change the v2 Host script. Mixed-exit Host is the **gap** book only.

Trades: `orb_asym_v2_pe_mix_trades.csv`.
