# Gap+VWAP 1-2-1 — Nifty vs Sensex, same-day vs expiry (2026-08-24)

**Calendar (from 1 Sep 2025):** Nifty weekly **Tuesday**, Sensex weekly **Thursday**.  
**Before that:** Nifty **Thursday**, Sensex **Tuesday**. Contracts through 31 Aug 2025 kept the old days.

Nifty search already used that switch (`EXPIRY_SWITCH = 2025-09-01`). Sensex now uses the **inverse**. Holiday expiry is still “next bar in cache”, not exchange previous-day rule.

**Question:** same gap 0.58–3.5% + 10:00 VWAP book, screenshot-style 1-2-1, hold 15:15 vs hold to weekly expiry.

Vehicle:
- Nifty: 50-pt grid, CE floor50 / +150 / +250, PE floor50−100 / −150 / −250, lot **65**
- Sensex: 100-pt analog, CE floor100 / +300 / +500, PE floor100−200 / −300 / −500, lot **10** (Dhan skill fallback; if lot is 20, double the ₹)

Sensex 5m: OpenAlgo `BSE_INDEX` / `SENSEX`, 2021-08-04 → 2026-06-30 (`sensex_5m_cache.pkl`). Volume present → VWAP is volume-weighted. Nifty cache volume is 0.

## Head-to-head

₹ = pts × lot. Sharpe still daily pts/50 (Nifty convention — Sensex pts are not the same unit).

| Book | Exit | n | WR | Sharpe | Avg pts | Avg ₹ | H1 WR/Sh | H2 WR/Sh |
|---|---|---:|---:|---:|---:|---:|---|---|
| **Nifty** | 15:15 | 106 | **61.3%** | **3.00** | +4.26 | **₹277** | 65.6 / 3.15 | 55.6 / 2.75 |
| **Nifty** | expiry | 106 | 51.9% | 3.37 | +9.10 | **₹592** | 50.8 / 3.44 | 53.3 / 3.25 |
| **Sensex** | 15:15 | 99 | 59.6% | **0.87** | +1.93 | **₹19** | 68.3 / 2.72 | **46.2 / −2.25** |
| **Sensex** | expiry | 99 | 52.5% | 2.42 | +12.04 | **₹120** | 56.7 / 3.82 | **46.2 / 0.0** |

Sensex same-day **fails Sharpe > 2** and **H2**. Expiry Sharpe 2.42 on the full sample; **H2 WR 46%** still fails. Rupees stay small because lot 10 vs 65.

## CE / PE

| | Nifty 15:15 | Nifty expiry | Sensex 15:15 | Sensex expiry |
|---|---|---|---|---|
| CE n / WR / avg | 65 / 66% / +4.05 | 65 / 65% / **+14.6** | 56 / 66% / +4.12 | 56 / 57% / +12.5 |
| PE n / WR / avg | 41 / **54%** / +4.59 | 41 / **32%** / +0.38 | 43 / 51% / **−0.93** | 43 / 47% / +11.5 |

Nifty expiry extra rupees = **up-gap calls**. Sensex expiry PE mean looks fine (+11.5) but WR is still <50%; same-day PE is negative.

## What this does *not* say

- Sensex is not a drop-in for the Nifty 15:15 lock.
- Platform max-profit (Nifty ~₹8k / unit) is still the right tail, not the mean.
- Sensex weekly options were thin / absent early in 2021–23 — index path is a proxy.
- MIS cannot hold to expiry on either index.

Trades: `orb_asym_expiry_hold_trades.csv` (Nifty) · `orb_asym_sensex_expiry_trades.csv`.  
Script: `orb_asym_v3_search.py expiry` / `sensex`.

**Next action:** keep the live hypothesis on **Nifty same-day 15:15** (or Nifty CE hold-to-Tuesday if you want rupees and will use NRML). Do not paper the Sensex clone of this book on current evidence.
