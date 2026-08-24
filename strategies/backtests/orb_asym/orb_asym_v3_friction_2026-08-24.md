# ORB_Asym v3 — friction overlay (2026-08-24)

**Question:** can the screenshot 1-2-1 still clear Sharpe > 2, WR > 50%, n ≥ 80, and H1/H2 after **flat 1–2 pt** cost?

Zero-cost primary (quiet LH/HL, OR 30–80, spot +30/−20, hard 14:30) is **n=187 WR 52.9 Sh 3.91 avg +0.93**. At 1 pt: WR 50.3 PF 0.96 Sh **−0.30**. That is the thing this pass beats.

Vehicle unchanged: CE floor50 / +150 / +250; PE floor50−100 / −150 / −250; strikes locked; BS Δnet σ=15%. LH/HL direction. No Defensive OR-Hold. No DTE-skip. Both CE and PE.

Script: `orb_asym_v3_search.py` (`friction_main`, `hunt2_main`).  
Grids: `orb_asym_v3_friction.csv`, `orb_asym_v3_friction2.csv`.  
Trades: `orb_asym_v3_friction_best_trades.csv`.

## Locked config (2 pt walk-forward; not deployed)

| Item | Value |
|---|---|
| Signals | 3-bar LH/HL, quiet \|prev net\| ≤ 0.42%, close vs OR mid, **first signal of the day only** |
| OR | 09:15–09:44, width **40–75**, skip range at 10:15 |
| Entry | 09:45–**11:00** |
| PE filter | skip PE if entry debit < **15** (CE unfiltered). Not a DTE skip. |
| Exit | trail: arm **+8** MTM, giveback **4**, stop **−8**; hard **15:15** |
| Mark | locked-strike BS Δnet minus flat cost |

CE 58 / PE 36. **0 credits.** Exits: HARD 39 · TRAIL 36 · MTM_STP 19.

| Cost | n | WR | PF | Sharpe | Avg | H1 WR/Sh (n=54) | H2 WR/Sh (n=40) |
|---|---:|---:|---:|---:|---:|---|---|
| 0 | 94 | 62.8% | 2.67 | 4.88 | **+4.05** | 63.0 / 4.33 | 62.5 / 5.96 |
| **1** | 94 | **60.6%** | **2.09** | **3.67** | **+3.05** | 59.3 / 3.26 | 62.5 / 4.50 |
| **2** | 94 | **56.4%** | **1.64** | **2.47** | **+2.05** | 55.6 / 2.18 | 57.5 / 3.03 |

Zero-cost avg **+4.05** vs old +0.93. After 2 pt, avg still **+2.05**.

## Simpler 1 pt passer (no first-only; did not survive 2 pt WF)

Quiet, OR 40–70, entry to 11:00, side MTM CE 25/12 PE 12/10, skip PE debit < 15, hard 14:30. n=106 (CE 61 / PE 45).

| Cost | WR | Sharpe | Avg | H1 WR/Sh | H2 WR/Sh |
|---|---:|---:|---:|---|---|
| 0 | 56.6 | 5.74 | +3.23 | 54.8 / 5.81 | 59.1 / 5.69 |
| 1 | 54.7 | 3.92 | +2.23 | 51.6 / 4.17 | 59.1 / 3.52 |
| 2 | 49.1 | 2.14 | +1.23 | 45.2 / 2.55 | 54.5 / 1.40 |

Kept as the less-filtered 1 pt book. 2 pt fails H1 WR and H2 Sharpe.

## What was skipped

- **DTE-skip** — still rejected.
- **debit_min = 25** on both legs — 2 pt numbers look good but PE collapses (e.g. 75 CE / 14 PE). Not both-sides.
- Relabelling **Spread** as Asym.
- Tiny 6 pt MTM scalps (high WR, no live book).
- Declaring 2 pt impossible after the first friction grid (948 one-pt WF passers, 0 two-pt). Second hunt (`first_only` + mid + OR 40–75 + trail) found 11 two-pt passers, all with `first_only` and PE floor 15.

## Caveats

- H2 is **n=40** — knife-edge vs a 40-trade half. One quiet year can move Sharpe/WR.
- Stacked filters (quiet + OR band + mid + first-only + PE floor). Overfit risk vs the n=187 zero-cost primary.
- Flat cost is not a chain: live 3-leg bid-ask can exceed 2 pts, especially on PE.
- Trail uses bar-close MTM; no fill slippage inside the bar.
- All 11 two-pt passers need **first-only**. Same-day bull+bear is off.

**Not deployed.** v2 Host script stays.

**Next action:** if papering v3, use this 2 pt lock (one unit, first signal only). If same-day both sides is mandatory, the 1 pt side-MTM book is the fallback and should not be expected to live at 2 pt friction.
