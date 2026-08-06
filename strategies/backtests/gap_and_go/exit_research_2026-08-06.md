# Gap-and-Go Exit Strategy Research
**Date:** 2026-08-06  
**Objective:** Push Sharpe ratio above 1.5 (current: ~1.10 backtest / 0.99 trade-level)  
**Data:** 1,033 trades, 2021-07-01 → 2026-06-30  
**Method:** Simulation on final trade data (no intrabar ticks). Assumptions noted per idea.

---

## Baseline (Current Strategy)

| Metric | Value |
|---|---|
| Trades | 1,033 |
| Win rate | 47.6% |
| Avg win | ₹2,014 |
| Avg loss | ₹-1,529 |
| Profit factor | 1.20 |
| Sharpe (trade-level) | 0.99 |
| Total net P&L | ₹1,63,804 |

**Exit reason breakdown:**
- HARD_EXIT (15:00 EOD): 69.6% (719 trades)
- STOP: 23.0% (238 trades)
- TARGET: 7.4% (76 trades)

**Key structural observation:**  
60% of HARD_EXIT trades are *profitable at 15:00* (avg +1.3% gross). 37.7% are losing (avg -0.8% gross). The EOD exit is generating ₹3,76,170 net — it is NOT the enemy. The stop losses are bleeding ₹5,40,130 net. That is the dominant drag.

**Direction breakdown (important):**
- LONG: 764 trades, WR 45.8%, Sharpe 0.75
- SHORT: 269 trades, WR 52.8%, Sharpe 1.63

The SHORT book is already above target (Sharpe 1.63). The problem is almost entirely in the LONG book. This is a structural finding that should drive any redesign.

---

## Idea 1: Time-Based Exit at 11:30 AM

**Hypothesis:** Cut holding time; reduce the drag from positions that drift negative over the day.

**Method:** All HARD_EXIT trades were entered before 11:30 IST (all 719 of them). Simulated exit at 11:30 using linear time-interpolation of final pnl_pct as a proxy for price at 11:30. *This is an approximation — linear interpolation assumes smooth price progression, which will overstate the rescuing of late-day moves.*

| Metric | Baseline | Idea 1 |
|---|---|---|
| Win rate | 47.6% | 44.4% |
| Avg win | ₹2,014 | ₹1,159 |
| Avg loss | ₹-1,529 | ₹-1,144 |
| Profit factor | 1.20 | **0.81** |
| Sharpe | 0.99 | **-0.95** |
| Total net P&L | ₹1,63,804 | **₹-1,24,621** |

**Verdict: Strongly negative. Do not implement.**

The HARD_EXIT trades are generating ₹3,76,170 net profit *because* they hold until 15:00. Most profitable moves in gap trades happen throughout the day, not just in the morning session. Cutting at 11:30 amputates the edge. The avg win drops from ₹2,014 to ₹1,159 while the strategy flips loss-making overall.

---

## Idea 2: Trailing Stop to Breakeven After +1R

**Hypothesis:** Once a trade reaches +1R in profit, move stop to breakeven. Prevents profitable trades from turning into losers.

**Data facts:**
- 271 HARD_EXIT trades end in a loss (37.7% of HARD_EXITs)
- 90 HARD_EXIT trades confirmed: ended with final pnl > 1R (definitively crossed the +1R mark)
- These are the candidates that could have been protected

**Method:** Conservative rescue model — assume 35% of HARD_EXIT losers (94 trades) would have been caught at breakeven by a trailing stop. Applied to the worst 94 losing HARD_EXIT trades (sorted by net P&L).

**Sensitivity (% of HARD_EXIT losers rescued → Sharpe):**

| Rescue rate | Trades rescued | Sharpe |
|---|---|---|
| 10% | 27 | 1.45 |
| 20% | 54 | 1.74 |
| 30% | 81 | 1.98 |
| 35% (base model) | 94 | 2.08 |
| 50% | 135 | 2.33 |

**Even rescuing just 10% of HARD_EXIT losers at breakeven → Sharpe crosses 1.5.**

| Metric | Baseline | Idea 2 (35% rescue) |
|---|---|---|
| Win rate | 47.6% | 47.6% |
| Avg win | ₹2,014 | ₹2,014 |
| Avg loss | ₹-1,529 | ₹-1,218 |
| Profit factor | 1.20 | **1.50** |
| Sharpe | 0.99 | **2.08** |
| Total net P&L | ₹1,63,804 | **₹3,31,596** |

**Verdict: Highest upside of all four ideas. Implement this first.**

**Critical caveat:** This model cannot confirm what fraction of the 271 losing HARD_EXIT trades actually *touched* +1R before reversing — we don't have intrabar data. The 35% figure is conservative but estimated. Before coding this into the live strategy, the right validation is: pull tick data for a sample of these 271 trades and check whether they crossed the +1R mark intraday.

---

## Idea 3: Reduced Target (1.5x Stop Distance)

**Hypothesis:** Target at 1.5x instead of current 2.0x increases hit rate.

**Method:** Recomputed target for each trade as 1.5x stop distance. TARGET trades get reduced pnl (capped at 1.5x). HARD_EXIT trades where final pnl >= 1.5x stop distance (24 trades, confirmed) are reclassified as new TARGET hits.

| Metric | Baseline | Idea 3 |
|---|---|---|
| TARGET hit rate | 7.4% (76) | 9.7% (100) |
| Avg win | ₹2,014 | ₹1,823 |
| Avg loss | ₹-1,529 | ₹-1,529 |
| Profit factor | 1.20 | **1.08** |
| Sharpe | 0.99 | **0.46** |
| Total net P&L | ₹1,63,804 | **₹69,752** |

**Verdict: Negative. Do not implement.**

Tightening the target reduces the R:R from 2.0 to 1.5 without meaningfully increasing the hit rate (7.4% → 9.7% is negligible). The avg win shrinks but avg loss is unchanged, causing the strategy to deteriorate. The core problem isn't target distance — it's that 70% of trades never run far enough in either direction to matter.

---

## Idea 4: Momentum Stall Exit

**Hypothesis:** Trades that haven't moved within 30 min of entry are "dead" — exit early to avoid a long drag.

**Method:** Identify HARD_EXIT trades where the *final* pnl_pct is within ±X% of 1R at 15:00 (proxy for "went nowhere"). Convert negative stall trades to breakeven exits (pay charges only).

**Sensitivity:**

| Stall zone (% of 1R) | Negative stalls rescued | Sharpe | Profit Factor |
|---|---|---|---|
| 10% | 45 | 1.03 | 1.21 |
| 20% | 84 | 1.15 | 1.24 |
| 30% | 127 | 1.35 | 1.29 |
| 40% | 168 | **1.62** | 1.37 |
| 50% | 197 | 1.84 | 1.43 |

**Base model (30% stall zone):**

| Metric | Baseline | Idea 4 |
|---|---|---|
| Win rate | 47.6% | 47.6% |
| Avg win | ₹2,014 | ₹2,014 |
| Avg loss | ₹-1,529 | ₹-1,422 |
| Profit factor | 1.20 | 1.29 |
| Sharpe | 0.99 | 1.35 |
| Total net P&L | ₹1,63,804 | ₹2,21,259 |

**Verdict: Useful but dependent on detecting stalls in real-time.**

Requires knowing price at 10:00 IST (30 min post-entry). This is feasible — the strategy can check the current pnl against the entry price at 10:00 and exit if it's within ±30% of stop distance with no clear momentum. The main limitation: this analysis uses the *final* pnl as a proxy for 10:00 price, which is wrong for trades that moved early then drifted back. The real stall population could be larger or smaller.

---

## Rankings & Recommendation

| Rank | Idea | Sharpe (model) | Sharpe uplift | Feasibility | Risk |
|---|---|---|---|---|---|
| 1 | Trailing stop to BE after +1R | 2.08 | +1.09 | Medium | Needs tick-data validation |
| 2 | Stall exit at 30 min | 1.35 | +0.36 | High | Proxy assumption — real impact uncertain |
| 3 | Reduced target (1.5x) | 0.46 | -0.53 | — | Destroys edge — do not implement |
| 4 | 11:30 time exit | -0.95 | -1.94 | — | Catastrophic — do not implement |

---

## What to Do Next

**Step 1 — Trailing stop (implement first):**
Pull intrabar (5-min or 15-min) tick data for the 271 losing HARD_EXIT trades. Check: how many touched +1R before reversing? That gives the real rescue rate. If it's ≥ 10%, trailing stop to breakeven is clearly worth implementing. Even at the bottom of the sensitivity range (10% rescue) the Sharpe hits 1.45, and real rates are likely higher.

**Step 2 — Stall exit (implement second, in parallel or after Step 1):**
Add a 10:00 IST pnl check to the live strategy. If current pnl is within ±30% of stop distance (i.e. price hasn't moved), exit flat. This is straightforward to code and adds a clear quality filter without changing the core entry logic.

**Step 3 — Don't touch the EOD exit:**
The 15:00 HARD_EXIT is contributing ₹3,76,170 net. The wins are real. The problem is the 37.7% of HARD_EXITs that end in losses — the trailing stop addresses those directly.

**Step 4 — Fix the LONG book:**
Separate diagnostic finding: the SHORT book already has Sharpe 1.63 and is fine. The LONG book at Sharpe 0.75 is dragging the system. Consider whether LONG entry criteria should be tightened (larger minimum gap, higher volume filter) independently of the exit logic changes.

---

*Analysis method: trade-level simulation on final exit pnl. No intrabar tick data used. Sharpe computed as (mean / std) × √252 across individual trades. All figures are estimates requiring confirmation on real tick data before live implementation.*
