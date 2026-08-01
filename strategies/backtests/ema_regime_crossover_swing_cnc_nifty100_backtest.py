"""
EMA Regime Crossover -- Swing CNC, Nifty 100 Universe Expansion Test
(2026-08-01, Karan-requested follow-on to the EMA Swing CNC Multi-Day Redesign)

CONTEXT: the deployed EMA Swing CNC strategy (live/paper since 2026-07-31,
`OpenAlgo/strategies/ema_regime_crossover_swing_cnc_signal.py`) runs on a
static high-beta 15-name universe (13 fetched). Its base config, "Run 3"
(`ema_regime_crossover_swing_cnc_concentration_v3_tightstop_backtest.py`),
scored Sharpe(net) 2.76 / PF(net) 1.67 / WR(net) 34.9% / maxDD(net) 18.1% on
169 trades, 2021-07-01->2026-06-30. A rolling-universe variant (monthly and
quarterly re-ranking on the same 15-name-sized pool) was already tested and
REJECTED 2026-07-31 (both cadences underperformed the static baseline on
Sharpe/PF -- see equities-system/strategies/ema_regime_crossover.md,
"Rolling-Universe Test for Swing CNC" section).

THIS SCRIPT asks a different question: not re-ranking within a small pool,
but genuinely WIDENING the pool -- what happens if the static universe is
expanded to the full Nifty 100 (Nifty 50 + Nifty Next 50, union of the two
lists already sourced 2026-07-17 in `universe_ranking.py`), holding every
other rule (regime/entry/exit/stop/trail/max-hold/sizing/concentration-cap/
charges) byte-for-byte identical to Run 3. This is ONE run, not a sweep --
report the honest result of dropping in the larger universe unchanged, per
Karan's brief. No parameter is tuned here.

UNIVERSE: union of `universe_ranking.py`'s NIFTY_50 (49 names) and
NIFTY_NEXT_50 (50 names) = 99 unique names (both lists already exclude
overlap with each other; sourced 2026-07-17 from tickertape.in). This vault
has repeated, documented data-gap issues with ADANIENSOL and CGPOWER
specifically (Dhan gapped-interior-chunk fetch errors, first hit
2026-07-17, recurring in every subsequent backtest that includes them) --
expected to fail again here. Other names may also fail; failures are
logged and excluded non-fatally, same convention as every other backtest
script in this vault (never silently drop without noting it).

CONCENTRATION CAP GROUP MAPPING -- extended from Run 3's Adani-only map to
cover real parent/promoter groups across the full Nifty 100 pool (not
forced groupings -- only where an actual shared promoter/parent exists):

- ADANI_GROUP: ADANIENT, ADANIPORTS, ADANIPOWER, ADANIENSOL, ADANIGREEN,
  AMBUJACEM (Adani acquired Ambuja Cements & ACC in 2022 -- a real,
  post-2022 addition to the group, not a stretch)
- TATA_GROUP: TATASTEEL, TCS, TATACONSUM, TMPV (Tata Motors Passenger
  Vehicles demerger entity), TITAN, TRENT, TATAPOWER, TMCV (Tata Motors
  Commercial Vehicles demerger entity), TATACAP, INDHOTEL (Taj Hotels,
  Tata Group)
- BAJAJ_GROUP: BAJAJFINSV, BAJAJ-AUTO, BAJFINANCE, BAJAJHLDNG
- BIRLA_GROUP (Aditya Birla Group): GRASIM, HINDALCO, ULTRACEMCO
- MAHINDRA_GROUP: M&M, TECHM (Tech Mahindra is part of the Mahindra Group)
- VEDANTA_GROUP: VEDL, HINDZINC (both Vedanta Ltd subsidiaries)
- LT_GROUP: LT, LTM (LTIMindtree is an L&T group company)
- HDFC_GROUP: HDFCBANK, HDFCLIFE, HDFCAMC (sister-listed entities under the
  same promoter umbrella post the HDFC Ltd/HDFC Bank merger)
- SBI_GROUP: SBIN, SBILIFE (State Bank Group)
- RELIANCE_GROUP: RELIANCE, JIOFIN (Jio Financial Services, Reliance/Ambani
  group)

Deliberately NOT grouped despite superficial similarity: government-owned
PSU banks (BANKBARODA, CANBK, PNB, UNIONBANK) and PSU energy/infra names
(NTPC, POWERGRID, COALINDIA, ONGC, IOC, BPCL, GAIL, RECLTD, PFC, IRFC,
MAZDOCK, HAL, BEL) are each separately managed, separately promoted
government entities -- not one conglomerate the way Tata/Adani/Birla/Bajaj
are. Also NOT grouped: JSWSTEEL and JINDALSTEL -- different branches of the
historical Jindal family (Sajjan Jindal vs. Naveen Jindal) that have run as
fully independent, separately-promoted corporate groups for decades; a real
distinction, not a coincidence of similar names. Every other name defaults
to its own singleton group via `group_of()`'s existing fallback, unchanged
from Run 3.

EVERYTHING ELSE held byte-for-byte identical to
`ema_regime_crossover_swing_cnc_concentration_v3_tightstop_backtest.py`:
daily EMA(200) regime, daily EMA(9)/EMA(20) crossover entry, LONG-only CNC,
1.5xATR initial stop, trailing stop arming at +1.0xATR then trailing at
2.2xATR, 40-trading-day max-hold backstop, MAX_CONCURRENT=6, MAX_PER_GROUP=2,
ASSUMED_LEVERAGE=1, capital_per_trade = ALLOCATED_CAPITAL / MAX_CONCURRENT,
ALLOCATED_CAPITAL=250,000, same CNC delivery charges model (incl. DP charge),
same 2021-07-01->2026-06-30 window.

DATA VOLUME: ~99 names of daily bars over 5 years is a much larger fetch
than Run 3's 13-name run. No additional rate-limiting/backoff is added
beyond what every other large-universe script in this vault already does
(a plain sequential per-symbol `client.history()` call, failure logged and
skipped non-fatally) -- neither `ema_regime_crossover_test8_high_beta_backtest.py`
nor `ema_regime_crossover_backtest_resized.py` (the two largest-universe
precedents in this campaign, ~59-60 names each) use retry/backoff logic
either, so this follows the same established pattern rather than inventing
a new one. Just budget for a longer wall-clock run.
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

# ---- Regime / entry (test #8 family, ported to daily bars) -- UNCHANGED FROM RUN 3 ----
REGIME_EMA = 200
ENTRY_FAST, ENTRY_SLOW = 9, 20
ATR_PERIOD = 14

# ---- Exit scheme (replaces 15:00 IST hard exit) -- UNCHANGED FROM RUN 3 ----
INITIAL_STOP_ATR_MULT = 1.5
TRAIL_ARM_ATR_MULT = 1.0     # favourable excursion needed to arm the trail
TRAIL_ATR_MULT = 2.2         # trailing distance once armed, same risk unit as the stop
MAX_HOLD_DAYS = 40           # trading-day backstop if nothing else triggers

LONG_ONLY = True  # CNC cannot short-sell in Indian cash equity

# ---- Sizing (CNC: no leverage) -- UNCHANGED FROM RUN 3 ----
ALLOCATED_CAPITAL = 250_000  # matches this strategy's existing paper-mode capital figure
MAX_CONCURRENT = 6
ASSUMED_LEVERAGE = 1  # CNC requires full payment -- no MIS-style leverage assumption

# ---- Concentration cap -- UNCHANGED FROM RUN 3 ----
MAX_PER_GROUP = 2

# ---- Universe: full Nifty 100 = union of universe_ranking.py's NIFTY_50 + NIFTY_NEXT_50 ----
# Sourced 2026-07-17 from tickertape.in (Nifty 50, Nifty Next 50 pages), same lists
# already used to build the 59-name intraday universe and MTF leverage survey.
NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "BAJAJFINSV", "AXISBANK", "BHARTIARTL",
    "BAJAJ-AUTO", "BAJFINANCE", "BEL", "ASIANPAINT", "DRREDDY", "EICHERMOT",
    "COALINDIA", "CIPLA", "HDFCBANK", "GRASIM", "HCLTECH", "HDFCLIFE", "HINDUNILVR",
    "ICICIBANK", "INFY", "HINDALCO", "INDIGO", "ITC", "JSWSTEEL", "M&M", "KOTAKBANK",
    "LT", "MARUTI", "MAXHEALTH", "NESTLEIND", "POWERGRID", "NTPC", "ONGC", "SBILIFE",
    "SHRIRAMFIN", "SBIN", "RELIANCE", "TATASTEEL", "SUNPHARMA", "TCS", "TECHM",
    "TATACONSUM", "TMPV", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO", "ETERNAL", "JIOFIN",
]

NIFTY_NEXT_50 = [
    "ABB", "ADANIPOWER", "AMBUJACEM", "ADANIENSOL", "ADANIGREEN", "BAJAJHLDNG",
    "BANKBARODA", "CGPOWER", "ZYDUSLIFE", "DLF", "BPCL", "CANBK", "DIVISLAB",
    "BRITANNIA", "CHOLAFIN", "CUMMINSIND", "DMART", "BOSCHLTD", "HAL", "GODREJCP",
    "GAIL", "HDFCAMC", "IOC", "HINDZINC", "INDHOTEL", "JINDALSTEL", "LTM",
    "UNITDSPR", "MUTHOOTFIN", "MOTHERSON", "RECLTD", "PIDILITIND", "PFC", "PNB",
    "TATAPOWER", "SOLARINDS", "SHREECEM", "SIEMENS", "UNIONBANK", "VEDL", "TVSMOTOR",
    "TORNTPHARM", "VBL", "MAZDOCK", "IRFC", "LODHA", "HYUNDAI", "ENRIN", "TATACAP",
    "TMCV",
]

# Union, de-duplicated, order preserved (Nifty 50 first, then Nifty Next 50).
_seen = set()
UNIVERSE = []
for _sym in NIFTY_50 + NIFTY_NEXT_50:
    if _sym not in _seen:
        _seen.add(_sym)
        UNIVERSE.append(_sym)

# Issuer/sector group map -- extended from Run 3's Adani-only map to cover real
# parent/promoter groups across the full Nifty 100 pool. See module docstring
# for the full rationale on each grouping decision and what was deliberately
# NOT grouped (PSU banks/energy, JSW vs Jindal Steel & Power).
GROUP_OF = {
    # Adani Group (incl. Ambuja Cements/ACC, acquired 2022)
    "ADANIENT": "ADANI_GROUP", "ADANIPORTS": "ADANI_GROUP", "ADANIPOWER": "ADANI_GROUP",
    "ADANIENSOL": "ADANI_GROUP", "ADANIGREEN": "ADANI_GROUP", "AMBUJACEM": "ADANI_GROUP",
    # Tata Group
    "TATASTEEL": "TATA_GROUP", "TCS": "TATA_GROUP", "TATACONSUM": "TATA_GROUP",
    "TMPV": "TATA_GROUP", "TITAN": "TATA_GROUP", "TRENT": "TATA_GROUP",
    "TATAPOWER": "TATA_GROUP", "TMCV": "TATA_GROUP", "TATACAP": "TATA_GROUP",
    "INDHOTEL": "TATA_GROUP",
    # Bajaj Group
    "BAJAJFINSV": "BAJAJ_GROUP", "BAJAJ-AUTO": "BAJAJ_GROUP", "BAJFINANCE": "BAJAJ_GROUP",
    "BAJAJHLDNG": "BAJAJ_GROUP",
    # Aditya Birla Group
    "GRASIM": "BIRLA_GROUP", "HINDALCO": "BIRLA_GROUP", "ULTRACEMCO": "BIRLA_GROUP",
    # Mahindra Group
    "M&M": "MAHINDRA_GROUP", "TECHM": "MAHINDRA_GROUP",
    # Vedanta Group
    "VEDL": "VEDANTA_GROUP", "HINDZINC": "VEDANTA_GROUP",
    # L&T Group
    "LT": "LT_GROUP", "LTM": "LT_GROUP",
    # HDFC Group
    "HDFCBANK": "HDFC_GROUP", "HDFCLIFE": "HDFC_GROUP", "HDFCAMC": "HDFC_GROUP",
    # State Bank Group
    "SBIN": "SBI_GROUP", "SBILIFE": "SBI_GROUP",
    # Reliance / Ambani Group
    "RELIANCE": "RELIANCE_GROUP", "JIOFIN": "RELIANCE_GROUP",
}
def group_of(symbol):
    return GROUP_OF.get(symbol, f"SINGLE_{symbol}")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "ema_regime_crossover_swing_cnc_nifty100")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- CNC (delivery) charges -- reused byte-for-byte from Run 3 ----
BROKERAGE_FLAT = 0.0
BROKERAGE_PCT = 0.0
STT_PCT_DELIVERY = 0.001            # 0.1%, both legs
EXCHANGE_PCT = 0.0000297            # NSE cash-segment rate, product-type-independent
SEBI_PCT = 0.000001
STAMP_DUTY_PCT_DELIVERY = 0.00015   # 0.015% on buy turnover
GST_PCT = 0.18
DP_CHARGE_PER_ISIN = 12.50          # Dhan DP charge, per ISIN per sell-out instruction (verified 2026-07-30)

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
    dp_charge = DP_CHARGE_PER_ISIN
    gst = GST_PCT * (brokerage + exchange_chg + sebi_chg + dp_charge)
    total = brokerage + stt + exchange_chg + sebi_chg + stamp_duty + gst + dp_charge
    return {"brokerage": round(brokerage, 2), "stt": round(stt, 2), "exchange_chg": round(exchange_chg, 2),
            "sebi_chg": round(sebi_chg, 2), "stamp_duty": round(stamp_duty, 2), "dp_charge": round(dp_charge, 2),
            "gst": round(gst, 2), "total_charges": round(total, 2)}


print(f"Universe: {len(UNIVERSE)} names (full Nifty 100, union of NIFTY_50+NIFTY_NEXT_50) | "
      f"LONG_ONLY={LONG_ONLY} | product=CNC | capital=Rs{ALLOCATED_CAPITAL:,} | "
      f"max_concurrent={MAX_CONCURRENT} | max_per_group={MAX_PER_GROUP} | "
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
            continue  # ran off the end of history with no exit -- excluded (END_OF_DATA), same convention as Run 3

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
# time before evaluating that candidate -- same convention as Run 3 and the
# intraday sibling's circuit-breaker backtest.
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
result_df.to_csv(os.path.join(OUT_DIR, "ema_regime_crossover_swing_cnc_nifty100_trades.csv"), index=False)

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

    print(f"\n=== EMA Regime Crossover -- Swing CNC, Nifty 100 Universe "
          f"(Rs{ALLOCATED_CAPITAL:,}, max {MAX_CONCURRENT} concurrent, max {MAX_PER_GROUP}/group) ===")
    print(f"Trades: {n} | WR (gross): {wr_gross:.1f}% | WR (net): {wr_net:.1f}%")
    print(f"Gross P&L: Rs {total_gross:+,.0f} | PF (gross): {pf_gross:.2f}")
    print(f"Charges  : Rs {total_charges:,.0f}  ({total_charges/total_gross*100 if total_gross else float('nan'):.1f}% of gross P&L)")
    print(f"Net P&L  : Rs {total_net:+,.0f} | PF (net): {pf_net:.2f}")
    print(f"Avg holding period: {result_df['hold_days'].mean():.1f} trading days "
          f"(median {result_df['hold_days'].median():.0f})")
    print(f"Exit breakdown: {result_df['reason'].value_counts().to_dict()}")
    print(f"By group (top 10): {result_df['group'].value_counts().head(10).to_dict()}")

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

    print(f"\nTrade log -> {OUT_DIR}/ema_regime_crossover_swing_cnc_nifty100_trades.csv")
