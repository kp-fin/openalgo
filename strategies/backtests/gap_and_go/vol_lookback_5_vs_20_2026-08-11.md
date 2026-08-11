# Gap-and-Go — Volume Lookback 5d vs 20d

**Date:** 2026-08-11  
**Question:** does shortening `VOL_LOOKBACK_DAYS` from 20 to 5 improve the deployed (LONG 3–10% filtered) config?  
**Method:** same script `gap_and_go_backtest.py`, same window 2021-07-01→2026-06-30, same 60-name universe, `GAG_FILTER_LONG=1`, only `GAG_VOL_LOOKBACK=5` changed.  
**Trade log:** `gap_and_go_trades_vol5_filtered.csv`

## Result

| Metric | 20d (deployed baseline) | 5d |
|---|---|---|
| Trades | 499 | 532 |
| WR (net) | 50.9% | 49.4% |
| PF (net) | **1.35** | 1.25 |
| Sharpe (net) | **1.62** | **1.26** |
| Net P&L | +₹138,859 | +₹102,530 |
| Max DD (net) | — | 15.0% |
| LONG n / PF(net) | 230 / — | 255 / 1.16 |
| SHORT n / PF(net) | 269 / — | 277 / 1.34 |

Volume-gate rejection count stayed almost flat (6,455 vs historical ~6,459), but the *which* days pass changed enough to add ~33 net trades of worse quality. Sharpe falls below the vault's >=1.5 goal.

**Caveat:** this 5d run also skipped TMCV (no data) in addition to the known CGPOWER/ADANIENSOL gaps — so 57 names vs the baseline's 58. One missing name cannot explain a Sharpe drop of 1.62→1.26.

## Verdict

**Rejected.** Keep `VOL_LOOKBACK_DAYS = 20`. Do not re-run 5 vs 20 without a new structural reason (e.g. joint retune of `VOL_MULT` with lookback).
