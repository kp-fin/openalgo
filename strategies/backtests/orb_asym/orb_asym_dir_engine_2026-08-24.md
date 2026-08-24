# Screenshot 1-2-1 — non-ORB direction engine (2026-08-24)

**Question:** can the screenshot fly make money if direction is **not** ORB (no LH/HL vs OR, no OR width, no skip-range, no Defensive OR-Hold)?

Vehicle locked: CE floor50 / +150 / +250; PE floor50−100 / −150 / −250; strikes locked at entry; BS Δnet σ=15%; ₹ ≈ pts × 65 × 1 unit. No DTE-skip. Host not touched.

Script: `OpenAlgo/strategies/backtests/orb_asym_v3_search.py` (`engine`, `selective`, `gap`, `gaphold`).  
Cache: `orb_asym/nifty_5m_cache.pkl` (volume is **always 0** — VWAP is equal-weight typical).

## Locked config (not deployed)

| Item | Value |
|---|---|
| Engine | **Gap continuation** + **session VWAP side at 10:00** |
| Gap | \|open / prev close − 1\| in **0.58%–3.5%**; trade the gap’s direction |
| Confirm | 10:00 close vs session VWAP (typical (H+L+C)/3, equal weight) |
| Entry | 10:00, one trade per day (CE on up-gap, PE on down-gap) |
| Exit | **Hold to 15:15** on fly Δnet (no intrinsic TARGET, no trail) |
| ORB | **Not used** |

CE 65 / PE 41. **0 credits.** All 106 exits HARD. Includes DTE=0 (18) — not skipped.

| Cost | n | WR | PF | Sharpe | Avg pts | Avg ₹ | H1 (n=61) WR/Sh | H2 (n=45) WR/Sh |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 106 | 61.3% | 1.92 | **3.00** | **+4.26** | **₹277** | 65.6 / 3.15 | 55.6 / 2.75 |
| 1 | 106 | 58.5% | 1.65 | **2.29** | **+3.26** | **₹212** | 63.9 / 2.52 | **51.1 / 1.92** |
| 2 | 106 | 55.7% | 1.41 | 1.59 | +2.26 | ₹147 | 62.3 / 1.88 | 46.7 / 1.09 |

Vs ORB friction lock avg **+4.05 pts (~₹263)** at 0 cost: this book is **+4.26 (~₹277)**. After 1 pt, avg **+3.26 (~₹212)** still beats **+2.05 (~₹133)**; overall WR/Sharpe still clear. **H2 Sharpe at 1 pt is 1.92** — misses the walk-forward Sharpe>2 by a hair. At 2 pt, avg still +2.26 but Sharpe 1.59 and H2 WR 46.7% fail.

### CE / PE (0 cost)

| Side | n | WR | Avg pts | Avg ₹ |
|---|---:|---:|---:|---:|
| CE | 65 | 66.2% | +4.05 | ₹263 |
| PE | 41 | 53.7% | +4.59 | ₹299 |

Not a CE-only story. PE win rate is weaker; average is not.

Trades: `orb_asym_dir_g58_35_1000_trades.csv`.  
Grid: `orb_asym_dir_gap_hold.csv`.

## Runner-up (higher 0-cost avg, worse 1 pt H2)

`g60_35_1000` hold (gap 0.60–3.5%, same VWAP confirm). n=99 CE 58 / PE 41. Avg **+4.46 pts (~₹290)** / Sh 3.10 / H2 n=42 WR 52.4 Sh 2.19. At 1 pt H2 WR **47.6%** Sh 1.38 — worse walk-forward than the lock. Trades: `orb_asym_dir_g60_35_1000_trades.csv`.

## What failed (rejection memory)

Do not retry without a new reason. None of these beat +4.05 **and** cleared H1/H2 Sharpe>2 at 0 cost except the gap+VWAP hold band above.

| Engine | Best 0-cost avg (both sides, n≥80) | Why rejected |
|---|---|---|
| Prior-day close vs open (09:45/10:00, ±0.3% band) | +0.49 | Dies at 1 pt; Sharpe < 1.1 |
| Gap continuation **unconfirmed** (0.15–0.30%, 09:45) | +0.48 | Same |
| Gap **fade** | −0.11 | Wrong way |
| VWAP side alone (10:00, dist≥20) | +0.53 | Dies at 1 pt |
| Opening drive vs open (09:45, \|Δ\|≥25) — **not** OR width | +0.67 (best lean-grid) | Best simple engine; still Sharpe ~1.0, dead after 1 pt |
| Yesterday H/L break after 09:45 | +0.46 | No edge |
| 15m HH/HL or LH/LL **without OR box** | +0.40 | Not ORB, also no edge |
| Prior-day **and** gap agree | +0.43 | Starves without helping |
| Drive + VWAP (no gap) | +1.09 | Below lock; 1 pt dead |
| Triple prior-day + drive + VWAP | +1.27 | H1/H2 split; not enough avg |
| Gap ≥0.8% unconfirmed, trail 10/4 | +2.77 | H2 Sharpe 1.13; does not beat +4.05 |
| Same gap+VWAP with **trail 8/4 or 8/5** | ~+2.0–2.4 | Clears 0-cost WF on some rows; **loses the avg war vs ORB +4.05** |
| Hard 14:30 on the lock set | hold avg +1.11 | Need the afternoon |

Not tested as “the strategy”: CE-only slices; DTE-skip; Defensive OR-Hold; relabelling ORB_Spread.

## Caveats

- H2 n=45 — same knife-edge class as the ORB lock’s n=40.
- Gap cap 3.5% was tuned to lift H2 sample vs a 2.0% cap (n2 36→45). Overfit risk.
- VWAP is not volume-weighted (cache volume=0).
- Hold-to-close has no stop: a 2 pt friction plus a bad afternoon wipes the 1 pt H2 gate.
- BS σ=15% is not a chain. Live 3-leg bid-ask can exceed 2 pts.
- 18 expiry-day (DTE=0) trades are in the book on purpose.

**Not deployed.** v2 Host script stays. This is not ORB_Spread.

**Next action:** if papering a non-ORB screenshot fly, use **gap 0.58–3.5% + 10:00 VWAP, hold 15:15**, one unit. Do not expect the 2 pt Sharpe gate. To clear 1 pt H2 Sharpe, need a stop that does not cut the hold winners — not found in this pass.
