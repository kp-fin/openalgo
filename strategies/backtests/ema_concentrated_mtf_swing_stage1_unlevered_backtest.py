"""
EMA Concentrated MTF Swing -- Stage 1 (UNLEVERED concentration test) (2026-08-01)

QUESTION BEING TESTED: per the design spec
(equities-system/strategies/ema_concentrated_mtf_swing.md), Karan wants 2-3
Nifty 100 names bought on MTF targeting Sharpe(net) >= 1.5, WR >= 40%. MTF
leverage was already tested and REJECTED on EMA Swing CNC's diversified
15-name book (raised P&L, worsened Sharpe/drawdown). Before touching leverage
again, Stage 1 isolates the ONE new variable this strategy actually
introduces: concentrating into a hard-screened top-2-3 instead of holding a
diversified 13-15-name book. This script is UNLEVERED (ASSUMED_LEVERAGE=1,
CNC-style charges) on purpose -- if concentration alone already fails to beat
EMA Swing CNC's own Sharpe(net) 2.76 / PF(net) 1.67 / WR(net) 34.9% / maxDD
18.1% baseline, layering leverage on top would only compound the same failure
mode already seen once. Stage 2 (MTF-levered) only runs if this clears the
Sharpe(net) >= 1.5 / WR >= 40% gate.

WHAT IS NEW vs. EMA Swing CNC Run 3 / Rolling Universe scripts (this is the
only thing under test -- entry/exit/stop/trail mechanics below are copied
unchanged from those scripts, same as every prior variant in this family):

1. UNIVERSE: full Nifty 100 pool (99 unique names, NIFTY_50 + NIFTY_NEXT_50),
   copied verbatim from ema_regime_crossover_swing_cnc_nifty100_backtest.py --
   no new pool invented.

2. TREND-QUALITY SCREEN (NEW, this script only): a name is only eligible for
   entry if, in addition to the existing EMA200 BULL regime gate:
     - ADX(14) > ADX_THRESHOLD (25) at the crossover bar -- confirms an
       actual trend, not a marginal EMA200 cross. Neither existing EMA-family
       variant uses ADX; added here specifically because concentrating into
       2-3 names means each pick has to be a real trend on its own, not "fine
       as 1-of-15."
   Wilder's ADX, standard 14-period smoothing (see compute_adx()).

3. RANKING / CONCENTRATION (NEW, this script only): among names passing BOTH
   gates (BULL regime + ADX>25) with a fresh bullish EMA9/20 crossover on a
   given bar, rank by:
     score = zscore(ADX_14) + zscore(ema9_slope_20d)
   where ema9_slope_20d is the % change in EMA(9) over the trailing 20
   trading days, and both terms are z-scored cross-sectionally across all
   names generating an eligible signal that trading day (not across the
   whole pool -- only candidates actually signaling are competing for the
   scarce slots). This is a first-draft weighting, explicitly flagged as
   unswept in the design spec -- not tuned here, only implemented so a
   baseline result exists to sweep from later.

   MAX_CONCURRENT = 3 (vs. 6 for the diversified variants) -- this IS the
   concentration lever under test. When more eligible candidates signal on
   the same day than there are free slots (capacity < MAX_CONCURRENT), the
   higher-scored candidate(s) fill the slot(s) first; the rest are rejected
   for that day only (not blacklisted -- they can re-signal and re-rank
   later). A currently-held name that would no longer rank in a hypothetical
   top-3 is NOT force-exited -- there is no periodic re-rank/rotation in this
   script (unlike the Rolling Universe variant's monthly universe refresh);
   ranking only ever gates which NEW candidate wins a free slot on entry day.
   This matches the existing family's "gate new entries only, never force-
   close an open position" convention (regime flips, group caps, and the
   Rolling Universe script's own membership rotation all follow this same
   rule) and keeps this test isolated to concentration/screening, not a
   second new mechanic (rotation) at the same time.

4. SIZING (adjusted for 3 slots instead of 6): capital_per_trade =
   ALLOCATED_CAPITAL / MAX_CONCURRENT, same formula as every other variant in
   this family, just recomputed for MAX_CONCURRENT=3 so each of the 2-3 names
   held gets a proportionally larger capital slice.

WHAT IS UNCHANGED FROM EMA SWING CNC RUN 3 / ROLLING UNIVERSE (copied
byte-for-byte, not re-derived): regime (daily EMA200), entry (daily EMA9/20
bullish cross while BULL), initial stop (1.5xATR), trail arm (1.0xATR)/trail
distance (2.2xATR), REVERSE_CROSS alternate exit, MAX_HOLD_DAYS=40,
LONG_ONLY/CNC-style charges, ALLOCATED_CAPITAL=Rs 2,50,000, ASSUMED_LEVERAGE=1
(UNLEVERED -- the entire point of Stage 1), MAX_PER_GROUP=2,
compute_charges_cnc(), same window (2021-07-01 -> 2026-06-30), same
daily-return Sharpe/drawdown convention, same Nifty-100 group map.

Data: OpenAlgo `/history`, daily bars ("D"), NSE cash equity.
"""

import heapq
import os
import warnings

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

# ---- Regime / entry (unchanged from EMA Swing CNC family) ----
REGIME_EMA = 200
ENTRY_FAST, ENTRY_SLOW = 9, 20
ATR_PERIOD = 14
ADX_PERIOD = 14

# ---- Trend-quality screen (NEW, this script only) ----
ADX_THRESHOLD = 25
EMA_SLOPE_LOOKBACK = 20  # trading days, for ema9_slope_20d

# ---- Exit scheme (unchanged from EMA Swing CNC family) ----
INITIAL_STOP_ATR_MULT = 1.5
TRAIL_ARM_ATR_MULT = 1.0
TRAIL_ATR_MULT = 2.2
MAX_HOLD_DAYS = 40

LONG_ONLY = True

# ---- Sizing (concurrency changed to 3 -- the concentration lever) ----
ALLOCATED_CAPITAL = 250_000
MAX_CONCURRENT = 3
ASSUMED_LEVERAGE = 1  # UNLEVERED -- Stage 1 by design, see module docstring

# ---- Concentration cap (issuer/sector group, unchanged mechanic) ----
MAX_PER_GROUP = 2

# ---- Universe: full Nifty 100 pool, copied verbatim from
# ema_regime_crossover_swing_cnc_nifty100_backtest.py ----
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

_seen = set()
UNIVERSE = []
for _sym in NIFTY_50 + NIFTY_NEXT_50:
    if _sym not in _seen:
        _seen.add(_sym)
        UNIVERSE.append(_sym)

# Issuer/sector group map -- copied verbatim from
# ema_regime_crossover_swing_cnc_nifty100_backtest.py.
GROUP_OF = {
    "ADANIENT": "ADANI_GROUP", "ADANIPORTS": "ADANI_GROUP", "ADANIPOWER": "ADANI_GROUP",
    "ADANIENSOL": "ADANI_GROUP", "ADANIGREEN": "ADANI_GROUP", "AMBUJACEM": "ADANI_GROUP",
    "TATASTEEL": "TATA_GROUP", "TCS": "TATA_GROUP", "TATACONSUM": "TATA_GROUP",
    "TMPV": "TATA_GROUP", "TITAN": "TATA_GROUP", "TRENT": "TATA_GROUP",
    "TATAPOWER": "TATA_GROUP", "TMCV": "TATA_GROUP", "TATACAP": "TATA_GROUP",
    "INDHOTEL": "TATA_GROUP",
    "BAJAJFINSV": "BAJAJ_GROUP", "BAJAJ-AUTO": "BAJAJ_GROUP", "BAJFINANCE": "BAJAJ_GROUP",
    "BAJAJHLDNG": "BAJAJ_GROUP",
    "GRASIM": "BIRLA_GROUP", "HINDALCO": "BIRLA_GROUP", "ULTRACEMCO": "BIRLA_GROUP",
    "M&M": "MAHINDRA_GROUP", "TECHM": "MAHINDRA_GROUP",
    "VEDL": "VEDANTA_GROUP", "HINDZINC": "VEDANTA_GROUP",
    "LT": "LT_GROUP", "LTM": "LT_GROUP",
    "HDFCBANK": "HDFC_GROUP", "HDFCLIFE": "HDFC_GROUP", "HDFCAMC": "HDFC_GROUP",
    "SBIN": "SBI_GROUP", "SBILIFE": "SBI_GROUP",
    "RELIANCE": "RELIANCE_GROUP", "JIOFIN": "RELIANCE_GROUP",
}
def group_of(symbol):
    return GROUP_OF.get(symbol, f"SINGLE_{symbol}")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "ema_concentrated_mtf_swing_stage1_unlevered")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- CNC (delivery) charges -- unchanged from the EMA Swing CNC family ----
BROKERAGE_FLAT = 0.0
BROKERAGE_PCT = 0.0
STT_PCT_DELIVERY = 0.001
EXCHANGE_PCT = 0.0000297
SEBI_PCT = 0.000001
STAMP_DUTY_PCT_DELIVERY = 0.00015
GST_PCT = 0.18
DP_CHARGE_PER_ISIN = 12.50

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


def compute_adx(df, period=ADX_PERIOD):
    """Wilder's ADX, standard 14-period smoothing."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_high, prev_low, prev_close = high.shift(1), low.shift(1), close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_wilder = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period).mean() / atr_wilder
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period).mean() / atr_wilder

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx


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


print(f"Universe: {len(UNIVERSE)} names (Nifty 100) | ADX threshold: {ADX_THRESHOLD} | "
      f"MAX_CONCURRENT={MAX_CONCURRENT} (concentration lever) | ASSUMED_LEVERAGE={ASSUMED_LEVERAGE} (unlevered) | "
      f"LONG_ONLY={LONG_ONLY} | product=CNC-style charges | capital=Rs{ALLOCATED_CAPITAL:,} | "
      f"max_per_group={MAX_PER_GROUP} | stop={INITIAL_STOP_ATR_MULT}xATR | trail_arm={TRAIL_ARM_ATR_MULT}xATR | "
      f"trail={TRAIL_ATR_MULT}xATR | max_hold={MAX_HOLD_DAYS}d")

price_data = {}
skipped = {}

for symbol in UNIVERSE:
    df, err = _fetch_daily(symbol, "NSE")
    if err:
        skipped[symbol] = err
        print(f"  {symbol}: SKIPPED ({err})")
        continue

    df["ema200"] = df["close"].ewm(span=REGIME_EMA, adjust=False, min_periods=REGIME_EMA).mean()
    df["regime"] = np.where(df["close"] > df["ema200"], "BULL", "BEAR")
    df["ema_fast"] = df["close"].ewm(span=ENTRY_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ENTRY_SLOW, adjust=False).mean()
    df["atr"] = compute_atr(df)
    df["adx"] = compute_adx(df)
    df["ema_fast_slope_20d"] = df["ema_fast"].pct_change(EMA_SLOPE_LOOKBACK) * 100
    df["bull_cross"] = (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)) & (df["ema_fast"] > df["ema_slow"])
    df["bear_cross"] = (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)) & (df["ema_fast"] < df["ema_slow"])
    price_data[symbol] = df
    print(f"  {symbol}: {len(df):,} daily bars ({df.index[0].date()} -> {df.index[-1].date()})")

# ---- Signal candidates: per-symbol trend-quality-gated crossover signals,
# with the ADX/regime score captured at the SIGNAL bar for later cross-
# sectional ranking (see Step 2/3 in the module docstring). ----
raw_signals = []  # one row per (symbol, signal_date) that passes the gates

for symbol, df in price_data.items():
    df2 = df.dropna(subset=["atr", "adx", "ema200", "ema_fast_slope_20d"])
    n = len(df2)
    for i in range(n - 1):
        row = df2.iloc[i]
        if not (row["bull_cross"] and row["regime"] == "BULL" and row["adx"] > ADX_THRESHOLD):
            continue
        raw_signals.append({
            "symbol": symbol, "signal_date": df2.index[i],
            "entry_idx": i + 1, "adx": row["adx"], "ema_slope": row["ema_fast_slope_20d"],
        })

if not raw_signals:
    print(f"\nNo signals passed the trend-quality screen. Skipped: {skipped}")
    raise SystemExit(0)

signals_df = pd.DataFrame(raw_signals)
print(f"\nSignal candidates passing BULL regime + ADX>{ADX_THRESHOLD} screen, full "
      f"{len(price_data)}-name pool: {len(signals_df)}")

# ---- Cross-sectional z-score ranking, PER SIGNAL DATE (only candidates
# actually signaling that day compete for the scarce slots -- see docstring
# point 3). A single-candidate day trivially "wins" its own rank. ----
def _zscore(s):
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std

signals_df["z_adx"] = signals_df.groupby("signal_date")["adx"].transform(_zscore)
signals_df["z_slope"] = signals_df.groupby("signal_date")["ema_slope"].transform(_zscore)
signals_df["score"] = signals_df["z_adx"] + signals_df["z_slope"]

# ---- Build full trade (entry->exit) for every screened+scored candidate;
# ranking/capacity constraints are applied at the portfolio-simulation stage
# below so an open position's own exit is never touched by a later day's
# ranking -- same "gate new entries only" convention as the rest of this
# strategy family. ----
all_trades = []
for _, sig in signals_df.iterrows():
    symbol = sig["symbol"]
    df2 = price_data[symbol].dropna(subset=["atr", "adx", "ema200", "ema_fast_slope_20d"])
    i = sig["entry_idx"]
    if i >= len(df2):
        continue
    entry_ts = df2.index[i]
    entry_bar = df2.iloc[i]
    entry_price = entry_bar["open"]
    atr_entry = df2.iloc[i - 1]["atr"]
    initial_stop = entry_price - INITIAL_STOP_ATR_MULT * atr_entry
    arm_level = entry_price + TRAIL_ARM_ATR_MULT * atr_entry

    trail_stop = initial_stop
    armed = False
    highest_close = entry_price
    exit_px, exit_ts, reason, hold_days = None, None, None, None
    rest = df2[df2.index > entry_ts]
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
        highest_close = max(highest_close, bar2["close"])
        if not armed and bar2["close"] >= arm_level:
            armed = True
        if armed:
            candidate_trail = highest_close - TRAIL_ATR_MULT * bar2["atr"]
            trail_stop = max(trail_stop, candidate_trail)
    if exit_px is None:
        continue

    pnl_pct = (exit_px - entry_price) / entry_price
    all_trades.append({
        "symbol": symbol, "direction": "LONG", "signal_date": sig["signal_date"],
        "score": sig["score"], "entry_time": entry_ts, "entry_price": entry_price,
        "exit_time": exit_ts, "exit_price": exit_px, "pnl_pct": pnl_pct,
        "reason": reason, "hold_days": hold_days,
    })

if not all_trades:
    print(f"\nNo trades generated from screened signals. Skipped: {skipped}")
    raise SystemExit(0)

trades_df = pd.DataFrame(all_trades).sort_values(["entry_time", "score"], ascending=[True, False]).reset_index(drop=True)
print(f"Trades built from screened signals (pre-portfolio-constraints): {len(trades_df)}")

# ---- Portfolio simulation: chronological event sim. On any entry_time where
# more candidates arrive than free slots, higher score wins first (score
# ordering already applied above within each entry_time group). ----
capital_per_trade = ALLOCATED_CAPITAL / MAX_CONCURRENT
buying_power = capital_per_trade * ASSUMED_LEVERAGE

pending_exits = []
seq = 0
open_positions = {}
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
result_df.to_csv(os.path.join(OUT_DIR, "ema_concentrated_mtf_swing_stage1_unlevered_trades.csv"), index=False)

print(f"\nSkipped (no data): {skipped}")
print(f"Rejected -- concurrency cap ({MAX_CONCURRENT}): {rejected_cap} | "
      f"concentration cap ({MAX_PER_GROUP}/group): {rejected_group}")

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

    print(f"\n=== EMA Concentrated MTF Swing -- Stage 1 (UNLEVERED) "
          f"(Rs{ALLOCATED_CAPITAL:,}, max {MAX_CONCURRENT} concurrent, max {MAX_PER_GROUP}/group) ===")
    print(f"Trades: {n} | WR (gross): {wr_gross:.1f}% | WR (net): {wr_net:.1f}%")
    print(f"Gross P&L: Rs {total_gross:+,.0f} | PF (gross): {pf_gross:.2f}")
    print(f"Charges  : Rs {total_charges:,.0f}  ({total_charges/total_gross*100 if total_gross else float('nan'):.1f}% of gross P&L)")
    print(f"Net P&L  : Rs {total_net:+,.0f} | PF (net): {pf_net:.2f}")
    print(f"Avg holding period: {result_df['hold_days'].mean():.1f} trading days "
          f"(median {result_df['hold_days'].median():.0f})")
    print(f"Exit breakdown: {result_df['reason'].value_counts().to_dict()}")
    print(f"By group: {result_df['group'].value_counts().to_dict()}")
    print(f"By symbol (top 15): {result_df['symbol'].value_counts().head(15).to_dict()}")

    print("\nBy exit reason (net of charges):")
    for reason, g in result_df.groupby("reason"):
        print(f"  {reason:14s} n={len(g):5d} ({len(g)/n*100:4.1f}%)  WR={(g.net_pnl_rupees>0).mean()*100:5.1f}%  "
              f"avg_pnl_pct={g.pnl_pct.mean()*100:+.3f}%  avg_hold={g.hold_days.mean():.1f}d  "
              f"total_net=Rs{g.net_pnl_rupees.sum():+10,.0f}")

    result_df["exit_time"] = pd.to_datetime(result_df["exit_time"])
    result_df["exit_date"] = result_df["exit_time"].dt.date
    daily_net = result_df.groupby("exit_date")["net_pnl_rupees"].sum() / ALLOCATED_CAPITAL
    daily_gross = result_df.groupby("exit_date")["pnl_rupees"].sum() / ALLOCATED_CAPITAL

    def sharpe(s):
        return s.mean() / s.std(ddof=1) * np.sqrt(252)

    print(f"\nSharpe (gross): {sharpe(daily_gross):.2f}")
    print(f"Sharpe (net)  : {sharpe(daily_net):.2f}   "
          f"(gate: >=1.5 | EMA Swing CNC diversified-15 baseline: 2.76)")

    dd_df = result_df.sort_values("exit_time").reset_index(drop=True)
    dd_df["cum_pnl_net"] = dd_df["net_pnl_rupees"].cumsum()
    dd_df["peak_net"] = dd_df["cum_pnl_net"].cummax()
    dd_df["dd_net"] = dd_df["cum_pnl_net"] - dd_df["peak_net"]
    max_dd = dd_df["dd_net"].min()
    print(f"\nMax drawdown (net): Rs {max_dd:,.0f} ({abs(max_dd)/ALLOCATED_CAPITAL*100:.1f}% of allocated capital)")
    print(f"WR (net) vs gate: {wr_net:.1f}% ({'PASS' if wr_net >= 40 else 'FAIL'} >=40% gate)")

    print(f"\nTrade log -> {OUT_DIR}/ema_concentrated_mtf_swing_stage1_unlevered_trades.csv")
