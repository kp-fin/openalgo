"""
EMA Regime Crossover -- Multi-Day/Swing CNC Redesign with Issuer-Concentration Cap
(2026-07-30, Karan-requested follow-on to the High-Beta Universe Test)

CONTEXT: every prior attempt on this strategy (the 17-test charges-aware
campaign, the 3 "v2" concepts, and the high-beta-universe test) kept the
15:00 IST intraday hard exit as an explicit, out-of-scope constraint. The
dominant failure mode across ALL of them was the same: a wide risk:reward
target essentially never completes before the hard exit, so trades resolve
as a same-day coin-flip while still paying full round-trip charges. This
script is the first to actually remove that constraint -- multi-day CNC
(delivery) holding, no same-day forced exit -- to test whether that
structural change (not another entry/exit tweak) closes the WR/Sharpe gap.

DESIGN CHOICES (all documented per Karan's brief):

1. Regime/entry logic reused: test #8's own EMA(200) regime + EMA(9)/EMA(20)
   crossover entry, THE best-performing entry/regime family found across the
   whole campaign (test #8 is the least-bad of 17+3 attempts, and its
   high-beta-universe variant is the only positive-net-Sharpe result seen at
   all). Ported from 30m-regime/15m-entry to DAILY bars throughout, since
   intraday-timeframe noise has no reason to matter once the position is
   meant to be held for days -- a same-day 15m EMA9/20 wobble is irrelevant
   information for a multi-day hold. This is a genuine adaptation, not a
   re-run of the intraday script on daily data: the daily EMA(200) regime
   here is a real ~200-trading-day (~10 month) trend filter, unlike the
   intraday version's misleadingly-named "16 trading day" EMA(200)-on-30m.
   REJECTED the v2 daily-EMA(50) alternative as the regime basis -- all
   three v2 concepts (including the ORB variant that used daily EMA50)
   underperformed test #8 on every metric, so there is no basis-with-evidence
   reason to start from EMA50 instead of the actually-best-performing EMA200
   family.

2. Product/direction: CNC (delivery), LONG-ONLY. CNC cannot short-sell in
   Indian cash equity (already established 2026-07-30 in the parked
   `ema_regime_crossover_backtest_narrow_window_long_only_cnc.py` attempt) --
   a genuine swing redesign is necessarily long-only on cash equity unless
   F&O (stock futures) is used for shorting. Defaulting to LONG-only CNC per
   Karan's already-indicated preference (same file, same day) rather than
   silently picking F&O shorting, which carries different margin/rollover/
   expiry mechanics not modeled anywhere in this vault yet.

3. Universe: the high-beta 15-name subset from the High-Beta Universe Test
   (2026-07-30), NOT the full ~59-60-name universe. Rationale: (a) it is the
   only sub-universe in the whole campaign shown empirically to lift
   average-win-per-trade relative to fixed per-trade costs (test #8: Sharpe
   net -0.24 -> +0.23 on this list), so it is the strongest available
   starting point for a redesign that is also trying to beat a charges
   floor; (b) it is small enough that the concentration cap below (part B
   of this task) actually gets exercised meaningfully rather than being a
   no-op on a 59-name universe where any one issuer group is a rounding
   error. Same 2 names failed to fetch previously (ADANIENSOL, CGPOWER) --
   expected to fail again here, effective universe ~13 names.

4. Concentration cap (NEW portfolio-level control, part B of this task):
   the high-beta list is heavily Adani-group-concentrated (5 of top 8 by
   volatility rank) -- flagged in the High-Beta Universe Test but never
   acted on. Generalized rule, not hardcoded to Adani: no more than
   `MAX_PER_GROUP` (2) names from the same issuer/sector GROUP may be held
   concurrently. Enforced as a real entry-rejection constraint in the
   simulation loop (a candidate entry is skipped -- not queued, not
   resized -- if accepting it would put open-position-count for that
   candidate's group above the cap), with a counter proving it actually
   fires.

5. Exit scheme (replaces the 15:00 hard exit):
   - Initial stop: entry - 2.2xATR(14, daily) -- same stop multiple as test
     #8's own (already-widened) stop, just on daily ATR instead of 15m ATR.
   - No fixed profit target. Once favourable excursion reaches
     `TRAIL_ARM_ATR_MULT` (1.5xATR) from entry, a trailing stop arms and
     ratchets up daily: `trail = highest_close_since_entry -
     TRAIL_ATR_MULT (2.2) x ATR(14, daily)`, recomputed each day (not frozen
     at arm time) -- lets a winning trend run rather than capping it, the
     same mechanism the existing (separate) EMA-Regime-Crossover-Swing
     sibling strategy already validated works well for multi-day holds
     (locked TRAIL_ATR_MULT=6.0 there, on a different regime/entry family --
     2.2 is used here instead to stay consistent with test #8's own risk
     unit rather than importing that strategy's separately-tuned value).
   - Alternate exit: opposite EMA(9)/EMA(20) daily crossover (REVERSE_CROSS),
     whichever comes first -- same convention as both intraday siblings.
   - Backstop: `MAX_HOLD_DAYS` (20 trading days) forced exit at that day's
     close if nothing else has triggered -- prevents an unbounded capital
     lock-up on a name that neither stops out nor reverses.

6. Sizing: CNC has no MIS-style assumed leverage (`ASSUMED_LEVERAGE = 1`) --
   full payment required, per the parked CNC attempt's own finding. Does
   NOT assume MTF (margin trading facility) leverage either -- MTF is a
   real, separate, per-symbol-metered product (see the sibling Swing
   strategy's own MTF leverage survey, 2.38x-4.55x per name) that this
   script deliberately does not invoke, since Karan has not asked for an
   MTF variant here and MTF margin economics (interest cost, per-symbol
   multiplier) are not modeled anywhere in this script. Capital is split
   evenly across `MAX_CONCURRENT` slots: `capital_per_trade =
   ALLOCATED_CAPITAL / MAX_CONCURRENT`, full slice used for quantity
   (no further leverage).

7. Charges: CNC (delivery) charges model, reused byte-for-byte in formula
   from the parked `ema_regime_crossover_backtest_narrow_window_long_only_cnc.py`
   attempt (same script that already worked out the intraday-vs-delivery
   charges delta and stated its sourcing/assumptions) -- NOT the intraday
   MIS charges model used everywhere else in this campaign. STT 0.1% on
   BOTH buy and sell legs (vs 0.025% sell-only for intraday), stamp duty
   0.015% on buy turnover (vs 0.003%), brokerage assumed Rs 0 (typical
   Indian discount-broker delivery pricing, NOT verified against Dhan's
   actual rate card -- flag if wrong), exchange/SEBI charges unchanged
   (not product-type-dependent). No same-day round-trip double-dip since
   these are now genuinely multi-day trades.

Data: OpenAlgo `/history`, daily bars ("D" interval), NSE cash equity,
2021-07-01 -> 2026-06-30, same window as every other backtest this
campaign for comparability.
"""

import heapq
import os
import warnings
from datetime import time as dtime

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")

API_KEY = os.getenv("OPENALGO_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENALGO_API_KEY environment variable before running.")
HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
START_DATE = "2021-07-01"
END_DATE = "2026-06-30"
IST = pytz.timezone("Asia/Kolkata")

# ---- Regime / entry (test #8 family, ported to daily bars) ----
REGIME_EMA = 200
ENTRY_FAST, ENTRY_SLOW = 9, 20
ATR_PERIOD = 14

# ---- Exit scheme (replaces 15:00 IST hard exit) ----
INITIAL_STOP_ATR_MULT = 2.2
TRAIL_ARM_ATR_MULT = 1.5     # favourable excursion needed to arm the trail
TRAIL_ATR_MULT = 2.2         # trailing distance once armed, same risk unit as the stop
MAX_HOLD_DAYS = 40           # trading-day backstop if nothing else triggers

LONG_ONLY = True  # CNC cannot short-sell in Indian cash equity

# ---- Sizing (CNC: no leverage) ----
ALLOCATED_CAPITAL = 250_000  # matches this strategy's existing paper-mode capital figure
MAX_CONCURRENT = 6
ASSUMED_LEVERAGE = 1  # CNC requires full payment -- no MIS-style leverage assumption

# ---- Concentration cap (NEW, part B of this task) ----
MAX_PER_GROUP = 2

# ---- Universe: high-beta 15-name subset (High-Beta Universe Test, 2026-07-30) ----
HIGH_BETA_15 = [
    "ADANIGREEN", "ADANIENSOL", "ADANIENT", "ADANIPOWER", "ETERNAL", "LODHA",
    "TMCV", "VEDL", "ADANIPORTS", "CGPOWER", "MAXHEALTH", "CANBK",
    "SHRIRAMFIN", "TRENT", "JIOFIN",
]
UNIVERSE = HIGH_BETA_15

# Issuer/sector group map -- generalizable, not hardcoded to "cap Adani only".
# Any name not explicitly listed falls into its own singleton group (never
# capped against anything else). Adani-group/adjacent names grouped together
# per the concentration-risk finding flagged (not fixed) in the High-Beta
# Universe Test: 5 of the top 8 by volatility rank are Adani-group names.
GROUP_OF = {
    "ADANIGREEN": "ADANI_GROUP",
    "ADANIENSOL": "ADANI_GROUP",
    "ADANIENT": "ADANI_GROUP",
    "ADANIPOWER": "ADANI_GROUP",
    "ADANIPORTS": "ADANI_GROUP",
}
def group_of(symbol):
    return GROUP_OF.get(symbol, f"SINGLE_{symbol}")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "ema_regime_crossover_swing_cnc_concentration_v2_maxhold40")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- CNC (delivery) charges -- reused from the parked CNC attempt script ----
BROKERAGE_FLAT = 0.0
BROKERAGE_PCT = 0.0
STT_PCT_DELIVERY = 0.001            # 0.1%, both legs
EXCHANGE_PCT = 0.0000297            # NSE cash-segment rate, product-type-independent
SEBI_PCT = 0.000001
STAMP_DUTY_PCT_DELIVERY = 0.00015   # 0.015% on buy turnover
GST_PCT = 0.18

from openalgo import api as openalgo_api
client = openalgo_api(api_key=API_KEY, host=HOST)


def _fetch_daily(symbol, exchange):
    try:
        resp = client.history(symbol=symbol, exchange=exchange, interval="D",
                               start_date=START_DATE, end_date=END_DATE)
    except Exception as e:
        return None, f"error: {e}"
    if isinstance(resp, dict):
        if resp.get("status") != "success":
            return None, f"api error: {resp.get('message', resp)}"
        df = pd.DataFrame(resp.get("data", []))
        if df.empty:
            return None, "no data"
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    else:
        df = resp
        if df is None or df.empty:
            return None, "no data"
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert(IST)
    df.index = df.index.normalize()
    return df.sort_index(), None


def compute_atr(df, period=ATR_PERIOD):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_charges_cnc(entry_price, exit_price, qty):
    buy_turnover = entry_price * qty
    sell_turnover = exit_price * qty
    total_turnover = buy_turnover + sell_turnover
    brokerage = BROKERAGE_FLAT * 2
    stt = STT_PCT_DELIVERY * total_turnover
    exchange_chg = EXCHANGE_PCT * total_turnover
    sebi_chg = SEBI_PCT * total_turnover
    stamp_duty = STAMP_DUTY_PCT_DELIVERY * buy_turnover
    gst = GST_PCT * (brokerage + exchange_chg + sebi_chg)
    total = brokerage + stt + exchange_chg + sebi_chg + stamp_duty + gst
    return {"brokerage": round(brokerage, 2), "stt": round(stt, 2), "exchange_chg": round(exchange_chg, 2),
            "sebi_chg": round(sebi_chg, 2), "stamp_duty": round(stamp_duty, 2), "gst": round(gst, 2),
            "total_charges": round(total, 2)}


print(f"Universe: {len(UNIVERSE)} names (high-beta subset) | LONG_ONLY={LONG_ONLY} | product=CNC | "
      f"capital=Rs{ALLOCATED_CAPITAL:,} | max_concurrent={MAX_CONCURRENT} | max_per_group={MAX_PER_GROUP} | "
      f"stop={INITIAL_STOP_ATR_MULT}xATR | trail_arm={TRAIL_ARM_ATR_MULT}xATR | trail={TRAIL_ATR_MULT}xATR | "
      f"max_hold={MAX_HOLD_DAYS}d")
all_trades = []
skipped = {}

for symbol in UNIVERSE:
    df, err = _fetch_daily(symbol, "NSE")
    if err:
        skipped[symbol] = err
        print(f"  {symbol}: SKIPPED ({skipped[symbol]})")
        continue

    df["ema200"] = df["close"].ewm(span=REGIME_EMA, adjust=False, min_periods=REGIME_EMA).mean()
    df["regime"] = np.where(df["close"] > df["ema200"], "BULL", "BEAR")
    df["ema_fast"] = df["close"].ewm(span=ENTRY_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ENTRY_SLOW, adjust=False).mean()
    df["atr"] = compute_atr(df)
    df["bull_cross"] = (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)) & (df["ema_fast"] > df["ema_slow"])
    df["bear_cross"] = (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)) & (df["ema_fast"] < df["ema_slow"])

    print(f"  {symbol}: {len(df):,} daily bars ({df.index[0].date()} -> {df.index[-1].date()})")

    df = df.dropna(subset=["atr", "ema200"])
    n = len(df)
    for i in range(n - 1):
        row = df.iloc[i]
        if not (row["bull_cross"] and row["regime"] == "BULL"):
            continue
        if not LONG_ONLY:
            continue  # (kept for structural symmetry; LONG_ONLY is always True here -- CNC constraint)

        entry_ts = df.index[i + 1]
        entry_bar = df.iloc[i + 1]
        entry_price = entry_bar["open"]
        atr_entry = row["atr"]
        initial_stop = entry_price - INITIAL_STOP_ATR_MULT * atr_entry
        arm_level = entry_price + TRAIL_ARM_ATR_MULT * atr_entry

        trail_stop = initial_stop
        armed = False
        highest_close = entry_price
        exit_px, exit_ts, reason, hold_days = None, None, None, None
        rest = df[df.index > entry_ts]
        for hold_idx, (ts2, bar2) in enumerate(rest.iterrows(), start=1):
            if bar2["low"] <= trail_stop:
                exit_px, exit_ts, reason, hold_days = trail_stop, ts2, ("TRAIL_STOP" if armed else "STOP"), hold_idx
                break
            if bar2["bear_cross"]:
                exit_px, exit_ts, reason, hold_days = bar2["close"], ts2, "REVERSE_CROSS", hold_idx
                break
            if hold_idx >= MAX_HOLD_DAYS:
                exit_px, exit_ts, reason, hold_days = bar2["close"], ts2, "MAX_HOLD", hold_idx
                break
            # Ratchet the trailing level using today's completed bar, effective from tomorrow only.
            highest_close = max(highest_close, bar2["close"])
            if not armed and bar2["close"] >= arm_level:
                armed = True
            if armed:
                candidate_trail = highest_close - TRAIL_ATR_MULT * bar2["atr"]
                trail_stop = max(trail_stop, candidate_trail)
        if exit_px is None:
            continue  # ran off the end of history with no exit -- excluded (END_OF_DATA), same convention as the sibling Swing strategy

        pnl_pct = (exit_px - entry_price) / entry_price
        all_trades.append({
            "symbol": symbol, "direction": "LONG", "entry_time": entry_ts,
            "entry_price": entry_price, "exit_time": exit_ts, "exit_price": exit_px,
            "pnl_pct": pnl_pct, "reason": reason, "hold_days": hold_days,
        })

if not all_trades:
    print(f"\nNo trades generated. Skipped: {skipped}")
    raise SystemExit(0)

trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
print(f"\nSignal candidates (pre-portfolio-constraints): {len(trades_df)}")

capital_per_trade = ALLOCATED_CAPITAL / MAX_CONCURRENT
buying_power = capital_per_trade * ASSUMED_LEVERAGE

# Chronological event simulation: process exits due before a candidate's entry
# time before evaluating that candidate, same convention as the intraday
# sibling's circuit-breaker backtest (exits must be "already closed" before
# testing whether a new entry fits within capacity/group constraints).
pending_exits = []
seq = 0
open_positions = {}  # trade_seq -> {"symbol":..., "group":...}
group_open_count = {}

accepted = []
rejected_cap, rejected_group = 0, 0

for _, cand in trades_df.iterrows():
    qty = int(buying_power // cand["entry_price"])
    if qty <= 0:
        continue
    pnl_rupees = qty * cand["entry_price"] * cand["pnl_pct"]

    while pending_exits and pending_exits[0][0] <= cand["entry_time"]:
        _, ex_seq, ex_group = heapq.heappop(pending_exits)
        group_open_count[ex_group] = group_open_count.get(ex_group, 1) - 1
        del open_positions[ex_seq]

    grp = group_of(cand["symbol"])
    if len(open_positions) >= MAX_CONCURRENT:
        rejected_cap += 1
        continue
    if group_open_count.get(grp, 0) >= MAX_PER_GROUP:
        rejected_group += 1
        continue

    t = cand.to_dict()
    t["qty"] = qty
    t["pnl_rupees"] = round(pnl_rupees, 2)
    charges = compute_charges_cnc(cand["entry_price"], cand["exit_price"], qty)
    t.update(charges)
    t["net_pnl_rupees"] = round(pnl_rupees - charges["total_charges"], 2)
    t["group"] = grp
    accepted.append(t)

    seq += 1
    open_positions[seq] = {"symbol": cand["symbol"], "group": grp}
    group_open_count[grp] = group_open_count.get(grp, 0) + 1
    heapq.heappush(pending_exits, (cand["exit_time"], seq, grp))

result_df = pd.DataFrame(accepted)
result_df.to_csv(os.path.join(OUT_DIR, "ema_regime_crossover_swing_cnc_concentration_v2_maxhold40_trades.csv"), index=False)

print(f"Skipped (no data): {skipped}")
print(f"Rejected -- concurrency cap: {rejected_cap} | concentration cap ({MAX_PER_GROUP}/group): {rejected_group}")

if result_df.empty:
    print("\nNo trades survived portfolio constraints.")
else:
    n = len(result_df)
    wr_gross = (result_df["pnl_rupees"] > 0).mean() * 100
    wr_net = (result_df["net_pnl_rupees"] > 0).mean() * 100
    total_gross = result_df["pnl_rupees"].sum()
    total_charges = result_df["total_charges"].sum()
    total_net = result_df["net_pnl_rupees"].sum()

    gw = result_df[result_df["pnl_rupees"] > 0]["pnl_rupees"].sum()
    gl = abs(result_df[result_df["pnl_rupees"] <= 0]["pnl_rupees"].sum())
    pf_gross = gw / gl if gl > 0 else float("inf")
    nw = result_df[result_df["net_pnl_rupees"] > 0]["net_pnl_rupees"].sum()
    nl = abs(result_df[result_df["net_pnl_rupees"] <= 0]["net_pnl_rupees"].sum())
    pf_net = nw / nl if nl > 0 else float("inf")

    print(f"\n=== EMA Regime Crossover -- Swing CNC + Concentration Cap "
          f"(Rs{ALLOCATED_CAPITAL:,}, max {MAX_CONCURRENT} concurrent, max {MAX_PER_GROUP}/group) ===")
    print(f"Trades: {n} | WR (gross): {wr_gross:.1f}% | WR (net): {wr_net:.1f}%")
    print(f"Gross P&L: Rs {total_gross:+,.0f} | PF (gross): {pf_gross:.2f}")
    print(f"Charges  : Rs {total_charges:,.0f}  ({total_charges/total_gross*100 if total_gross else float('nan'):.1f}% of gross P&L)")
    print(f"Net P&L  : Rs {total_net:+,.0f} | PF (net): {pf_net:.2f}")
    print(f"Avg holding period: {result_df['hold_days'].mean():.1f} trading days "
          f"(median {result_df['hold_days'].median():.0f})")
    print(f"Exit breakdown: {result_df['reason'].value_counts().to_dict()}")
    print(f"By group: {result_df['group'].value_counts().to_dict()}")

    print("\nBy exit reason (net of charges):")
    for reason, g in result_df.groupby("reason"):
        print(f"  {reason:14s} n={len(g):5d} ({len(g)/n*100:4.1f}%)  WR={(g.net_pnl_rupees>0).mean()*100:5.1f}%  "
              f"avg_pnl_pct={g.pnl_pct.mean()*100:+.3f}%  avg_hold={g.hold_days.mean():.1f}d  "
              f"total_net=Rs{g.net_pnl_rupees.sum():+10,.0f}")

    # Daily-return Sharpe: allocate net P&L to its exit date, same convention
    # as every other backtest in this campaign, for direct comparability.
    result_df["exit_time"] = pd.to_datetime(result_df["exit_time"])
    result_df["exit_date"] = result_df["exit_time"].dt.date
    daily_net = result_df.groupby("exit_date")["net_pnl_rupees"].sum() / ALLOCATED_CAPITAL
    daily_gross = result_df.groupby("exit_date")["pnl_rupees"].sum() / ALLOCATED_CAPITAL

    def sharpe(s):
        return s.mean() / s.std(ddof=1) * np.sqrt(252)

    print(f"\nSharpe (gross): {sharpe(daily_gross):.2f}")
    print(f"Sharpe (net)  : {sharpe(daily_net):.2f}   (goal: >= 1.50)")

    dd_df = result_df.sort_values("exit_time").reset_index(drop=True)
    dd_df["cum_pnl_net"] = dd_df["net_pnl_rupees"].cumsum()
    dd_df["peak_net"] = dd_df["cum_pnl_net"].cummax()
    dd_df["dd_net"] = dd_df["cum_pnl_net"] - dd_df["peak_net"]
    max_dd = dd_df["dd_net"].min()
    print(f"\nMax drawdown (net): Rs {max_dd:,.0f} ({abs(max_dd)/ALLOCATED_CAPITAL*100:.1f}% of allocated capital)")

    print(f"\nTrade log -> {OUT_DIR}/ema_regime_crossover_swing_cnc_concentration_v2_maxhold40_trades.csv")
