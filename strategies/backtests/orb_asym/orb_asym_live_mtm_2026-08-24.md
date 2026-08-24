# ORB_Asym — live-like BS MTM vs locked intrinsic (2026-08-24)

**Question:** replay locked v2 entries/exits with a **1-2-1 option mark** (what live P&L uses), not intrinsic − 7.5.

**Method:** Black-Scholes European, 50pt strikes, weekly DTE calendar, σ ∈ {10,12,15,20}%. Same 255 trades / same spot TARGET 50 / STOP 25. P&L = `exit_net − entry_net`. Not a chain; no bid-ask.

Scripts: `orb_asym_live_mtm_backtest.py`. Trades: `orb_asym_live_mtm_trades.csv`. Grid: `orb_asym_live_mtm_summary.csv`.

## Headline (σ = 15%)

| Slice | Model | n | WR | PF | Sharpe | Net pts | Credit at entry |
|---|---|---:|---:|---:|---:|---:|---:|
| All | Intrinsic − 7.5 (locked) | 255 | 50.6% | 5.71 | 12.77 | **+4,401** | 0% |
| All | BS MTM | 255 | 76.5% | 4.94 | 6.85 | **+223** | **96%** |
| TARGET | Intrinsic − 7.5 | 122 | 100% | inf | — | **+5,185** (+42.5 avg) | 0% |
| TARGET | BS MTM | 122 | 100% | inf | — | **+97** (**+0.80 avg**) | 94% |

**TARGET does not pay ~42 pts on the fly.** It pays **~0.8 pts** under flat-IV BS (~₹52 / 65-share unit). Locked v2 overstates TARGET by **~50×**.

## Match to 21–24 Aug paper

- BS entries are **credits** (net0 typically −6 to −10), same sign as live (−0.5 to −13).
- STOP can print **positive ₹** when spot goes against (shorts cheapen) — 21 Aug.
- 24 Aug TARGET **lost** 3.75 pts. BS TARGET never went negative (min +0.07). Live was **worse** than BS: IV/skew on the tested body, three-leg spread, 1 DTE dump. Direction of the gap (spot win ≠ 42 pt fly) is robust; exact −₹244 is not in this model.

## Verdict

Live is not “broken TARGET logic.” The **vehicle mark** is a small, often-credit fly. Do **not** go live expecting v2 rupees. Paper TARGET ₹ is the evidence that matters.

**Not adopted:** change to v2 frozen settings. This is a model-gap finding, not a new config.
