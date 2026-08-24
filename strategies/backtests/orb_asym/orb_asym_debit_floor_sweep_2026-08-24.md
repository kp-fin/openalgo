# ORB_Asym — debit-floor / DTE sweep (2026-08-24)

**Question:** among debit-floor specs (live net > 0 / ≥3 / ≥7.5), which is most beneficial on the locked v2 paths?

**Honest limit:** v2 is **intrinsic − constant 7.5**. Every historical trade is assumed to pay the same debit. A live **entry skip** on quoted net cannot be replayed. Grid is therefore (a) **cost overlay** on all 255 trades, (b) **DTE skip** as the proxy for 1-DTE/credit-fly sessions (24 Aug was T−1 weekly).

Locked paths: `orb_asym_v2_best_trades.csv` (hold_quiet, body 50 / far 150, stop 25). Expiry weekday: Thu until 2025-08-31, Tue from 2025-09-01. CSV: `orb_asym_debit_floor_sweep.csv`.

## DTE mix (n=255)

| DTE | Trades |
|---:|---:|
| 0 (expiry day) | 66 |
| 1 | 53 |
| 2–7 | 136 |

T−1 + expiry = **119 trades (47%)**.

## Results (gates: Sharpe > 2.5, WR ≥ 45%, n ≥ 30)

| Spec | n | skipped | WR | PF | Sharpe | Net pts | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| Cost overlay debit=0 | 255 | 0 | 51.4 | inf | 18.58 | +6,313 | ✓ *not a floor* |
| Cost overlay debit=3 | 255 | 0 | 51.4 | 15.91 | 16.26 | +5,548 | ✓ *not a floor* |
| Locked debit=7.5, no skip | **255** | 0 | **50.6** | **5.71** | **12.77** | **+4,401** | ✓ **best real spec** |
| Skip expiry (DTE=0) | 189 | 66 | 48.1 | 5.17 | 11.71 | +3,017 | ✓ worse |
| Skip DTE≤1 | 136 | 119 | 45.6 | 4.61 | 10.77 | +1,965 | ✓ barely |
| Skip DTE≤1, H2 2024+ | 67 | — | **43.3** | 4.50 | 10.12 | +959 | **WR fail** |

Cost overlay “wins” by subtracting less from the same intrinsic. That **rewards cheaper assumed cost**; it does **not** test skipping credits. Live credits are the opposite of debit=0.

## Verdict

**Do not add a DTE/T−1 skip** — it cuts almost half the sample and fails H2 WR. **Do not pick debit=0 from this grid.**

Live **debit floor remains untested on history**. It is still a **risk control** vs three paper credits, not a backtest winner. If added, **≥3 pts** is operational (kills credits/near-zero), not 7.5 (would likely blank many live prints; not measured here).

**Frozen v2 stays:** debit assumption 7.5, no DTE filter, spot ±50/−25.
