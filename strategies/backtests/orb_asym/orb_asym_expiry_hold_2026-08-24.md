# Screenshot 1-2-1 — hold to weekly expiry vs 15:15 (2026-08-24)

**Question:** on the locked gap+VWAP book (`g58_35_1000`), does holding the fly to **weekly expiry** beat same-day **15:15**?

Same 106 entries. Strikes locked. Entry mark = BS Δnet (σ=15%). Expiry mark = **intrinsic** at expiry-day 15:15 close (T ≈ 0). ₹ = pts × 65 × 1 unit. No DTE-skip. Not deployed.

Script: `orb_asym_v3_search.py expiry`. Trades: `orb_asym_expiry_hold_trades.csv`.

## Result

| Exit | n | WR | PF | Sharpe | Avg pts | Avg ₹ | H1 WR/Sh | H2 WR/Sh |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Same-day 15:15 | 106 | **61.3%** | 1.92 | 3.00 | +4.26 | ₹277 | 65.6 / 3.15 | 55.6 / 2.75 |
| Hold to expiry | 106 | 51.9% | 1.71 | **3.37** | **+9.10** | **₹592** | 50.8 / 3.44 | 53.3 / 3.25 |
| Expiry −1 pt | 106 | 51.9% | 1.61 | 3.00 | +8.10 | ₹526 | 50.8 / 3.09 | 53.3 / 2.84 |
| Expiry −2 pt | 106 | 51.9% | 1.51 | 2.63 | +7.10 | ₹462 | 50.8 / 2.75 | 53.3 / 2.42 |

Headline: **higher average, worse win rate.** Gates WR>50% / Sharpe>2 still clear on the mixed book. H1 WR **50.8%** is on the wire.

18 trades are already DTE=0 (same clock as 15:15). Overnight slice:

| Exit | n | WR | Sharpe | Avg pts | Avg ₹ |
|---|---:|---:|---:|---:|---:|
| Same-day, DTE>0 | 88 | **62.5%** | 2.60 | +1.80 | ₹117 |
| Expiry, DTE>0 | 88 | 51.1% | 2.89 | +7.63 | ₹496 |

Overnight is where the extra rupees come from — and where win rate falls.

## Not the ₹8,287 screen

Expiry P&L: min **−42.6**, median **+3.8**, p90 **+84.6**, max **+120.6 pts (~₹7,839)**.  
14 / 106 trades >80 pts. 51 / 106 lose. The platform max is the **right tail**, not the mean.

## CE vs PE (this is the tell)

| Side | n | WR 15:15 | WR expiry | Avg 15:15 | Avg expiry |
|---|---:|---:|---:|---:|---:|
| CE (up-gap) | 65 | 66.2% | **64.6%** | +4.05 | **+14.59 (~₹948)** |
| PE (down-gap) | 41 | **53.7%** | **31.7%** | **+4.59** | **+0.38 (~₹25)** |

Expiry hold is a **call-fly / up-gap** book. Down-gap puts **lose the same-day edge** if you wait for Thursday. Do not treat +9.10 as a both-sides result.

## Caveats

- Intrinsic at 15:15 is not settlement / not a chain. No weekend gap inside the week except calendar DTE.
- Host is **MIS**. Holding to weekly expiry is **NRML**, different margin and carry.
- PE median expiry P&L **−22.7 pts**. Mean is rescued by a few winners.
- H1 WR 50.8% after expiry — one quiet year flips the gate.
- Live 3-leg exit on expiry day is worse than 2 pts of flat cost.

**Not deployed.** Same-day 15:15 remains the locked non-ORB exit unless you explicitly drop PE or accept ~32% put WR.

**Next action:** if the goal is rupees, expiry-hold **CE-only** is the hypothesis to test (n=65, not a full book). If the goal is WR, keep 15:15.
