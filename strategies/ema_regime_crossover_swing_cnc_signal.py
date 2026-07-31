"""
EMA Regime Crossover — Swing CNC (Concentration-Capped) — 15-name high-beta
universe — OpenAlgo Live/Paper Script (built 2026-07-30, NOT YET DEPLOYED)

Separate strategy from BOTH existing siblings:
  - ema_regime_crossover_signal.py           (intraday MIS, "test #8", 30m regime/15m entry)
  - ema_regime_crossover_swing_signal.py      (swing MTF, JdK RRG regime, retired 2026-07-27)
This is the "Multi-Day/Swing CNC Redesign + Concentration Cap" variant documented in
equities-system/strategies/ema_regime_crossover.md (2026-07-30 section of that name).
It ports Run 3 of that redesign campaign -- the leading verified candidate
(Sharpe(net) 2.76, PF(net) 1.67, WR(net) 34.9%, over 2021-07-01->2026-06-30, 169 trades)
-- EXACTLY, parameter-for-parameter, from the backtest that IS this spec byte-for-byte
on trading logic: strategies/backtests/ema_regime_crossover_swing_cnc_concentration_v3_tightstop_backtest.py.
Do not change INITIAL_STOP_ATR_MULT / TRAIL_ARM_ATR_MULT / TRAIL_ATR_MULT / MAX_HOLD_DAYS
/ the universe / the concentration-cap logic without re-running that backtest first --
frozen-strategy-settings rule applies here same as the deployed intraday sibling.

Regime + entry (test #8's own family, ported to DAILY bars -- same daily port already
used by Run 1/2/3 of the redesign, NOT the intraday sibling's 30m/15m timeframes and NOT
the JdK RRG regime the OTHER swing sibling uses):
  Regime: price above/below EMA(200) on the DAILY chart -> BULL/BEAR. LONG entries
          only permitted in BULL regime (CNC cannot short -- see Product below).
  Entry:  daily EMA(9) crosses above EMA(20) while in BULL regime -> LONG.
  Signal is read from the LAST COMPLETED daily bar only (today's still-forming bar is
  dropped before any indicator is computed -- same date-aware fix the RRG swing sibling
  applied 2026-07-22, ported here directly since the same live-vs-backtest completed-bar
  hazard applies to any daily-bar script that runs during market hours). Entry price is
  live LTP at the time of the once-daily signal pass, approximating the backtest's
  entry-at-next-open convention (script runs once shortly after open).

Exit (replaces the intraday sibling's 15:00 hard exit entirely -- there is no same-day
forced exit here, a position can run for days to weeks):
  - Initial stop: entry - 1.5xATR(14, daily).
  - Trailing stop: arms once favourable excursion reaches entry + 1.0xATR(14, daily)
    (TRAIL_ARM_ATR_MULT), then trails at highest_close_since_entry - 2.2xATR(14, daily)
    (TRAIL_ATR_MULT), recomputed once per day from that day's completed bar, ratchets up
    only (never loosens). Matches the backtest's own ratchet-from-next-day-only sequencing.
  - Alternate exit: opposite EMA(9)/EMA(20) daily crossover (REVERSE_CROSS).
  - Backstop: MAX_HOLD_DAYS (40 trading days) forced exit if nothing else has triggered,
    counted from entry_time using a days-held counter persisted in state (incremented once
    per daily signal pass, not per poll).
  Whichever of stop/trail/reverse-cross/max-hold triggers first wins, matching the
  backtest's per-bar precedence (stop/trail check before reverse-cross before max-hold).

Product/direction: CNC (cash & carry / delivery), LONG-ONLY -- CNC cannot short-sell
Indian cash equity (already established in the parked long-only-CNC backtest attempt
this spec's docstring references). NOT MIS (no same-day square-off -- this is a
multi-day hold) and NOT MTF (the separate RRG swing sibling's own mechanism, with its
own margin survey/interest-cost modeling -- explicitly tested here 2026-07-30 as
`ema_regime_crossover_swing_mtf_leverage_backtest.py` and REJECTED: real MTF leverage
raises absolute net P&L but makes both Sharpe(net) [2.76->2.19] and max drawdown
[18.1%->75.2%] materially worse on the same own-capital base, because MTF interest is a
fixed daily drag on the borrowed notional. ASSUMED_LEVERAGE stays 1 in this script,
deliberately, per that finding).

Sizing: NO leverage. capital_per_trade = allocated_capital / MAX_CONCURRENT (6), the
full resulting slice used for quantity -- no leverage multiplier layered on top (unlike
BOTH existing siblings, which apply ASSUMED_MIS_LEVERAGE / real MTF leverage on top of
their own capital_per_trade). This is the exact sizing formula Run 3's backtest used
(capital_per_trade = ALLOCATED_CAPITAL / MAX_CONCURRENT, buying_power = capital_per_trade
* ASSUMED_LEVERAGE with ASSUMED_LEVERAGE=1).

Universe: the 15-name high-beta subset from the High-Beta Universe Test (2026-07-30),
NOT either sibling's 20/59-name universe -- see HIGH_BETA_15 below. ADANIENSOL and
CGPOWER have historically failed to fetch from Dhan/OpenAlgo (confirmed in the backtest,
skipped[symbol] populated for both) -- this script logs a warning and continues without
them rather than treating a fetch failure as fatal; effective universe is ~13 names most
days. This is a STATIC list, no monthly ADV re-ranking (unlike both existing siblings'
refresh_universe() mechanism) -- the High-Beta Universe Test that produced this list was
itself a one-off ranking exercise, not an ongoing monthly process, and re-deriving it
here would be a new, unverified mechanism outside this task's scope.

Concentration cap (NEW vs. both existing siblings): no more than MAX_PER_GROUP (2)
concurrently-OPEN positions from the same issuer/sector GROUP (see GROUP_OF below --
generalized, not hardcoded to only ever check Adani; any symbol not listed defaults to
its own singleton group and is never capped against anything else). A candidate entry
that would push its group's open-count above the cap is skipped outright (not queued,
not resized) -- same enforcement the backtest already validated (10-11 rejections/run
across Run 1-3, confirmed actually binding).

Charges: a genuine CNC delivery charges model (compute_charges_cnc below, byte-for-byte
port of the backtest's own function, INCLUDING the DP charge Rs 12.50/ISIN/sell-out --
verified against Dhan's real rate card 2026-07-30, see the strategy doc) is computed and
logged on every closed trade for ongoing Sharpe/PF/WR tracking purposes ONLY. It does
NOT gate whether a trade is taken -- the broker (Dhan, via OpenAlgo) computes the real
charges at execution; this script's charges figure is an estimate for the trade log, not
a live cost feed.

Scheduling / cadence (CONFIRMED by Karan 2026-07-30): the DAILY signal pass (regime/entry/
bar-close stop-trail-ratchet/reverse-cross/max-hold logic) runs ONCE PER TRADING DAY at
09:15 IST -- Karan's explicit choice, overriding this script's original 09:20 placeholder
recommendation (no reason given to prefer waiting for opening-auction settling; 09:15 is
market open). Confirmed/added same day: an INTRADAY LTP safety check on the initial/
trailing stop between daily bar-close passes, protecting against an adverse intraday move
that blows through the stop and reverses by the next day's close (a real gap the pure
daily-bar backtest never modeled -- the backtest only ever checks the stop once per day at
close, so a same-day intraday spike-and-recovery below the stop is invisible to it; this
script's live behaviour is intentionally MORE conservative than its own backtest here).

Mechanics: this is still a SINGLE script/schedule, not two separate scheduled jobs --
schedule it to run repeatedly through the trading day (recommended every 15 minutes,
09:15-15:25 IST, matching the RRG swing sibling's own intraday cadence) via the Python
Strategy Host's scheduler. The FIRST invocation each day (when `state["last_signal_date"]
!= today`) runs the full daily signal pass exactly as before (`main()`/`_daily_pass()`).
EVERY SUBSEQUENT invocation that same day runs `_intraday_stop_check()` instead: for each
open position, fetch a live LTP and exit immediately (SELL at market, `reason=
"STOP_INTRADAY"` or `"TRAIL_STOP_INTRADAY"` depending on `armed`) if breached -- this does
NOT ratchet the trailing stop, bump `days_held`, or evaluate REVERSE_CROSS/MAX_HOLD (those
stay strictly bar-close/once-daily, exactly as before); it is purely a safety net against
the stop level already set by the last daily pass. No new entries are ever taken outside
the once-daily pass. If the intraday check exits a position, the position is simply gone
from state by the time the next daily pass runs -- no double-exit risk, no special-casing
needed between the two code paths beyond the shared state file.

Capital: reads allocated_capital via capital_state.get_allocated_capital("ema_regime_crossover_swing_cnc")
-- its OWN key, added to capital_allocation.json this same session (paper mode, Rs
2,50,000 starting, matching the backtest's own ALLOCATED_CAPITAL assumption) -- entirely
separate from the intraday sibling's "ema_regime_crossover" key and the RRG swing
sibling's "ema_regime_crossover_swing" key. Live-mode closed trades compound into this
key's own allocated_capital via capital_state.record_trade(); paper-mode trades never
touch it -- identical mechanism to both siblings.

State persisted to state/ema_regime_crossover_swing_cnc_state.json -- ENTIRELY SEPARATE
from both siblings' own state files, per this task's explicit requirement that this
script must not share state with, or interfere with, the already-running intraday
ema_regime_crossover_signal.py. Schema: positions (symbol -> entry_price, qty,
initial_stop, trail_stop, highest_close, atr_at_entry, days_held, armed, entry_time,
exit_order_placed/pending_reverse_exit retry flags -- same retry-safety pattern as both
existing siblings), group_open_count (group -> count, rebuilt from positions on load so
it can never drift out of sync with the actual open-position set), last_signal_date
(gates the once-per-day signal pass the same way both siblings gate their own daily/
5-minute passes).

NOT YET DEPLOYED: this script is NOT registered in strategy_configs.json and the Python
Strategy Host has NOT been started or scheduled with it. Building/validating it (compiles
clean, read-only smoke test against live OpenAlgo endpoints) is in scope for this task;
actually scheduling it live or in Sandbox paper mode is Karan's separate go-ahead, exactly
matching how the intraday sibling itself sat built-but-undeployed from 2026-07-17 to
2026-07-30 pending a separate decision.
"""

import json
import logging
import os
import sys
import time
from datetime import date, datetime, time as dtime, timedelta

import numpy as np
import pandas as pd
import pytz
import requests

# Deployed copies run from strategies/scripts/ (Python Strategy Host), where
# only the script's own directory is on sys.path by default -- add the parent
# strategies/ dir so capital_state.py resolves from both source and deployed
# locations without needing a duplicated copy kept in sync.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capital_state import get_allocated_capital, record_trade

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ema_regime_crossover_swing_cnc")

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("OPENALGO_API_KEY", "your_openalgo_api_key_here")
HOST    = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
IST     = pytz.timezone("Asia/Kolkata")

REGIME_EMA = 200
ENTRY_FAST = 9
ENTRY_SLOW = 20
ATR_PERIOD = 14

# ---- Exit scheme -- Run 3 of the swing-CNC redesign, locked 2026-07-30. Do not
# change without re-running ema_regime_crossover_swing_cnc_concentration_v3_tightstop_backtest.py ----
INITIAL_STOP_ATR_MULT = 1.5
TRAIL_ARM_ATR_MULT     = 1.0
TRAIL_ATR_MULT         = 2.2
MAX_HOLD_DAYS          = 40

LONG_ONLY = True   # CNC cannot short-sell Indian cash equity

# ---- Sizing (CNC: no leverage, no MTF -- see module docstring, MTF explicitly rejected) ----
MAX_CONCURRENT    = 6
ASSUMED_LEVERAGE  = 1   # locked -- do not add MTF/MIS-style leverage to this variant

# ---- Concentration cap ----
MAX_PER_GROUP = 2

# ---- Universe: high-beta 15-name subset (High-Beta Universe Test, 2026-07-30) ----
HIGH_BETA_15 = [
    "ADANIGREEN", "ADANIENSOL", "ADANIENT", "ADANIPOWER", "ETERNAL", "LODHA",
    "TMCV", "VEDL", "ADANIPORTS", "CGPOWER", "MAXHEALTH", "CANBK",
    "SHRIRAMFIN", "TRENT", "JIOFIN",
]
UNIVERSE = HIGH_BETA_15

# Issuer/sector group map -- generalizable, not hardcoded to "cap Adani only".
# Any name not explicitly listed falls into its own singleton group.
GROUP_OF = {
    "ADANIGREEN": "ADANI_GROUP",
    "ADANIENSOL": "ADANI_GROUP",
    "ADANIENT": "ADANI_GROUP",
    "ADANIPOWER": "ADANI_GROUP",
    "ADANIPORTS": "ADANI_GROUP",
}


def group_of(symbol):
    return GROUP_OF.get(symbol, f"SINGLE_{symbol}")


DAILY_LOOKBACK_DAYS = 400  # comfortably covers 200-day EMA warmup with buffer

STRATEGY_KEY = "ema_regime_crossover_swing_cnc"
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state",
                          "ema_regime_crossover_swing_cnc_state.json")
os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

# ---- CNC (delivery) charges -- byte-for-byte port of the backtest's compute_charges_cnc,
# now including the Dhan DP charge (verified 2026-07-30). LOGGING/TRACKING ONLY -- never
# gates whether a trade is taken; the broker computes the real charges at execution. ----
BROKERAGE_FLAT           = 0.0
STT_PCT_DELIVERY         = 0.001      # 0.1%, both legs
EXCHANGE_PCT             = 0.0000297  # NSE cash-segment rate, product-type-independent
SEBI_PCT                 = 0.000001
STAMP_DUTY_PCT_DELIVERY  = 0.00015    # 0.015% on buy turnover
GST_PCT                  = 0.18
DP_CHARGE_PER_ISIN       = 12.50      # Dhan DP charge, per ISIN per sell-out instruction


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


# ── State helpers ─────────────────────────────────────────────────────────────
def load_state():
    """Multi-day-hold strategy: positions persist across days, nothing resets
    daily -- same convention as the RRG swing sibling, not the intraday
    sibling's daily flatten. group_open_count is rebuilt from positions on
    every load rather than persisted independently, so it can never drift out
    of sync with the actual open-position set."""
    if os.path.exists(STATE_FILE):
        try:
            s = json.load(open(STATE_FILE))
        except Exception:
            s = {}
    else:
        s = {}
    s.setdefault("positions", {})
    s.setdefault("last_signal_date", None)
    s["group_open_count"] = {}
    for sym in s["positions"]:
        grp = group_of(sym)
        s["group_open_count"][grp] = s["group_open_count"].get(grp, 0) + 1
    return s


def save_state(s):
    # group_open_count is derived, not persisted -- drop it before writing so
    # it's always rebuilt fresh from positions on the next load.
    to_write = {k: v for k, v in s.items() if k != "group_open_count"}
    json.dump(to_write, open(STATE_FILE, "w"), default=str)


# ── OpenAlgo helpers ──────────────────────────────────────────────────────────
def _headers():
    return {"Content-Type": "application/json"}


def get_candles_daily(symbol, exchange="NSE", lookback_days=DAILY_LOOKBACK_DAYS):
    end   = date.today().isoformat()
    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    try:
        r = requests.post(f"{HOST}/api/v1/history",
                          json={"apikey": API_KEY, "symbol": symbol, "exchange": exchange,
                                "interval": "D", "start_date": start, "end_date": end},
                          headers=_headers(), timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "success" or not data.get("data"):
            log.warning(f"{symbol}: daily history fetch failed (continuing without it, not fatal): {data}")
            return pd.DataFrame()
        df = pd.DataFrame(data["data"])
        df.columns = [c.lower() for c in df.columns]
        if "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize()
        else:
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
        df = df.set_index("datetime").sort_index()
        return df
    except Exception as e:
        log.warning(f"{symbol}: daily history fetch error (continuing without it, not fatal): {e}")
        return pd.DataFrame()


def get_multiquotes(symbols):
    """symbols: list of symbol strings (all NSE). Returns {symbol: ltp}."""
    r = requests.post(f"{HOST}/api/v1/multiquotes",
                      json={"apikey": API_KEY,
                            "symbols": [{"symbol": s, "exchange": "NSE"} for s in symbols]},
                      headers=_headers(), timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(data)
    return {res["symbol"]: float(res["data"]["ltp"]) for res in data["results"]}


def place_order(symbol, action, quantity):
    r = requests.post(f"{HOST}/api/v1/placeorder",
                      json={"apikey": API_KEY, "strategy": STRATEGY_KEY, "symbol": symbol,
                            "exchange": "NSE", "action": action, "quantity": quantity,
                            "pricetype": "MARKET", "product": "CNC"},
                      headers=_headers(), timeout=15)
    r.raise_for_status()
    return r.json()


def log_closed_trade(sym, pos, exit_px, pnl, reason, now):
    """Append to this strategy's OWN trade-log CSV (paper AND live) -- separate
    from both siblings' logs (strategies/logs/ema_regime_crossover_swing_cnc_trades.csv,
    via capital_state.py's per-strategy-key file naming). Live mode also
    compounds pnl into this key's OWN allocated_capital."""
    charges = compute_charges_cnc(pos["entry_price"], exit_px, pos["qty"])
    net_pnl = pnl - charges["total_charges"]
    record_trade(STRATEGY_KEY, {
        "date": now.strftime("%Y-%m-%d"),
        "symbol": sym,
        "direction": "LONG",
        "entry_time": pos["entry_time"],
        "entry_price": round(pos["entry_price"], 2),
        "exit_time": now.isoformat(),
        "exit_price": round(exit_px, 2),
        "qty": pos["qty"],
        "pnl_rupees": round(pnl, 2),
        "charges_rupees": charges["total_charges"],
        "net_pnl_rupees": round(net_pnl, 2),
        "reason": reason,
        "hold_days": pos.get("days_held", 0),
        "group": group_of(sym),
    }, pnl)  # capital_state compounds the GROSS pnl (matches both siblings' own
             # convention -- charges are logged/tracked here, real charges are
             # whatever Dhan actually deducts, not double-subtracted from allocated_capital)


# ── Indicator helpers ────────────────────────────────────────────────────────
def compute_atr(df, period=ATR_PERIOD):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_indicators(symbol):
    """Daily OHLC + EMA200 regime + EMA9/20 cross + ATR14. Returns None if
    there's not enough history for a valid regime reading yet.

    COMPLETED BARS ONLY: the /history API returns TODAY's daily candle live and
    updating throughout the session -- today's bar (if present) is dropped
    before anything is computed, so iloc[-1] downstream is always the last
    genuinely completed bar, matching the backtest's completed-bar-then-next-
    open convention. Same fix the RRG swing sibling applied 2026-07-22."""
    today_ts = pd.Timestamp(date.today())
    df = get_candles_daily(symbol)
    if not df.empty:
        df = df[df.index < today_ts]
    if df.empty or len(df) < REGIME_EMA + 5:
        return None

    df = df.copy()
    df["ema200"] = df["close"].ewm(span=REGIME_EMA, adjust=False, min_periods=REGIME_EMA).mean()
    df["ema_fast"] = df["close"].ewm(span=ENTRY_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ENTRY_SLOW, adjust=False).mean()
    df["atr"] = compute_atr(df)
    df["regime"] = np.where(df["close"] > df["ema200"], "BULL", "BEAR")
    df["bull_cross"] = (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)) & (df["ema_fast"] > df["ema_slow"])
    df["bear_cross"] = (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)) & (df["ema_fast"] < df["ema_slow"])

    df = df.dropna(subset=["atr", "ema200"])
    return df if not df.empty else None


def _intraday_stop_check(state, now):
    """Safety-net check between daily passes (see module docstring "Scheduling /
    cadence", confirmed 2026-07-30). Checks live LTP against each open position's
    CURRENT stop/trail_stop (as last set by the daily bar-close pass) and exits
    immediately on a breach -- does NOT ratchet the trail, bump days_held, or
    evaluate REVERSE_CROSS/MAX_HOLD (those remain strictly once-daily/bar-close).
    A pure safety net, not a second signal pass."""
    positions = state["positions"]
    group_open_count = state["group_open_count"]
    if not positions:
        log.info("Intraday stop check: no open positions.")
        return

    try:
        ltps = get_multiquotes(list(positions.keys()))
    except Exception as e:
        log.error(f"Intraday stop check: quote fetch failed, skipping this pass: {e}")
        return

    for sym, pos in list(positions.items()):
        ltp = ltps.get(sym)
        if ltp is None:
            continue
        if ltp > pos["trail_stop"]:
            continue  # not breached -- nothing to do

        exit_reason = "TRAIL_STOP_INTRADAY" if pos.get("armed") else "STOP_INTRADAY"
        try:
            if not pos.get("exit_order_placed"):
                place_order(sym, "SELL", pos["qty"])
                pos["exit_order_placed"] = True  # a retry must never re-send the order
            pnl = (ltp - pos["entry_price"]) * pos["qty"]
            log.info(f"EXIT {exit_reason} {sym} entry={pos['entry_price']:.2f} exit={ltp:.2f} "
                     f"stop={pos['trail_stop']:.2f} pnl={pnl:+.0f} hold={pos.get('days_held', 0)}d")
            log_closed_trade(sym, pos, ltp, pnl, exit_reason, now)
            del positions[sym]
            grp = group_of(sym)
            group_open_count[grp] = max(0, group_open_count.get(grp, 1) - 1)
        except Exception as e:
            log.error(f"Intraday exit failed for {sym} ({exit_reason}, will retry next pass): {e}")

    save_state(state)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    """Dispatches to the once-daily signal pass or the intraday stop-check
    safety net, depending on whether today's daily pass has already run (see
    module docstring "Scheduling / cadence", confirmed 2026-07-30: schedule
    this script every ~15 minutes, 09:15-15:25 IST -- the first invocation each
    day runs the full daily pass, every subsequent one runs the intraday check
    only)."""
    now = datetime.now(IST)
    today_str = date.today().isoformat()
    state = load_state()

    if state.get("last_signal_date") == today_str:
        log.info(f"Intraday stop-check pass — {now.strftime('%Y-%m-%d %H:%M:%S IST')}")
        _intraday_stop_check(state, now)
        return

    log.info(f"EMA Regime Crossover Swing CNC daily signal pass — {now.strftime('%Y-%m-%d %H:%M:%S IST')}")

    allocated_capital = get_allocated_capital(STRATEGY_KEY)
    capital_per_trade  = allocated_capital / MAX_CONCURRENT   # NO leverage -- see module docstring "Sizing"

    positions = state["positions"]
    group_open_count = state["group_open_count"]

    symbols_to_check = sorted(set(UNIVERSE) | set(positions.keys()))
    indicators = {}
    for sym in symbols_to_check:
        ind = compute_indicators(sym)
        if ind is not None:
            indicators[sym] = ind
        else:
            log.warning(f"{sym}: no usable indicator data this pass (fetch failure or insufficient "
                        f"history) — continuing without it, not fatal (expected for ADANIENSOL/CGPOWER).")
        time.sleep(0.1)

    # ── Existing positions: stop/trail-stop check (bar-close basis, matches the
    #    backtest), then REVERSE_CROSS, then MAX_HOLD_DAYS backstop ───────────
    for sym, pos in list(positions.items()):
        ind = indicators.get(sym)
        if ind is None or ind.empty:
            continue
        last = ind.iloc[-1]  # last COMPLETED daily bar
        low, close = float(last["low"]), float(last["close"])
        atr = float(last["atr"])

        exit_reason = None
        if low <= pos["trail_stop"]:
            exit_reason = "TRAIL_STOP" if pos.get("armed") else "STOP"
        elif bool(last["bear_cross"]) or pos.get("pending_reverse_exit"):
            exit_reason = "REVERSE_CROSS"
        elif pos.get("days_held", 0) + 1 >= MAX_HOLD_DAYS:
            exit_reason = "MAX_HOLD"

        if exit_reason is not None:
            try:
                exit_px = get_multiquotes([sym]).get(sym)
                if exit_px is None:
                    # Never exit blind / log a fabricated fill -- persist a
                    # pending-exit flag and retry next signal pass, same
                    # pattern as both existing siblings.
                    if exit_reason == "REVERSE_CROSS":
                        pos["pending_reverse_exit"] = True
                    log.warning(f"{sym}: {exit_reason} triggered but no live price — "
                                f"will retry next signal pass")
                    continue
                if not pos.get("exit_order_placed"):
                    place_order(sym, "SELL", pos["qty"])
                    pos["exit_order_placed"] = True  # a retry must never re-send the order
                pnl = (exit_px - pos["entry_price"]) * pos["qty"]
                log.info(f"EXIT {exit_reason} {sym} entry={pos['entry_price']:.2f} exit={exit_px:.2f} "
                         f"pnl={pnl:+.0f} hold={pos.get('days_held', 0)}d")
                log_closed_trade(sym, pos, exit_px, pnl, exit_reason, now)
                del positions[sym]
                grp = group_of(sym)
                group_open_count[grp] = max(0, group_open_count.get(grp, 1) - 1)
            except Exception as e:
                if exit_reason == "REVERSE_CROSS":
                    pos["pending_reverse_exit"] = True
                log.error(f"Exit failed for {sym} ({exit_reason}, will retry next pass): {e}")
            continue

        # No exit -- ratchet the trailing stop and bump the hold counter.
        pos["days_held"] = pos.get("days_held", 0) + 1
        pos["highest_close"] = max(pos.get("highest_close", pos["entry_price"]), close)
        if not pos.get("armed") and close >= pos["arm_level"]:
            pos["armed"] = True
            log.info(f"{sym}: trailing stop ARMED at close={close:.2f} (arm_level={pos['arm_level']:.2f})")
        if pos.get("armed"):
            candidate_trail = pos["highest_close"] - TRAIL_ATR_MULT * atr
            pos["trail_stop"] = max(pos["trail_stop"], candidate_trail)
        log.info(f"HOLD LONG {sym} entry={pos['entry_price']:.2f} close={close:.2f} "
                 f"trail={pos['trail_stop']:.2f} armed={pos.get('armed', False)} "
                 f"hold={pos['days_held']}d")

    # ── New entries: bull_cross + BULL regime, concurrency cap, concentration
    #    cap, no existing position in that symbol ─────────────────────────────
    for sym in UNIVERSE:
        if len(positions) >= MAX_CONCURRENT:
            log.info(f"At {MAX_CONCURRENT}-position cap — no more new entries today.")
            break
        if sym in positions:
            continue
        ind = indicators.get(sym)
        if ind is None or ind.empty:
            continue
        last = ind.iloc[-1]
        if not (bool(last["bull_cross"]) and last["regime"] == "BULL"):
            continue
        if not LONG_ONLY:
            continue  # structural symmetry only -- LONG_ONLY is always True (CNC constraint)

        grp = group_of(sym)
        if group_open_count.get(grp, 0) >= MAX_PER_GROUP:
            log.info(f"SKIP {sym} — concentration cap ({MAX_PER_GROUP}/group) reached for group {grp}")
            continue

        atr = float(last["atr"])
        if atr <= 0 or np.isnan(atr):
            continue

        try:
            entry_price = get_multiquotes([sym]).get(sym)
        except Exception as e:
            log.error(f"Entry LTP fetch failed for {sym}: {e}")
            continue
        if entry_price is None:
            continue

        buying_power = capital_per_trade * ASSUMED_LEVERAGE   # ASSUMED_LEVERAGE=1, locked -- see module docstring
        qty = int(buying_power // entry_price)
        if qty <= 0:
            continue

        initial_stop = entry_price - INITIAL_STOP_ATR_MULT * atr
        arm_level = entry_price + TRAIL_ARM_ATR_MULT * atr

        try:
            place_order(sym, "BUY", qty)
            positions[sym] = {
                "qty": qty, "entry_price": entry_price, "entry_time": now.isoformat(),
                "initial_stop": initial_stop, "trail_stop": initial_stop,
                "arm_level": arm_level, "armed": False,
                "highest_close": entry_price, "days_held": 0,
                "atr_at_entry": atr,
            }
            group_open_count[grp] = group_open_count.get(grp, 0) + 1
            log.info(f"ENTRY LONG {sym} qty={qty} entry={entry_price:.2f} stop={initial_stop:.2f} "
                     f"arm_level={arm_level:.2f} atr={atr:.2f} group={grp}")
        except Exception as e:
            log.error(f"Entry failed for {sym}: {e}")

    state["last_signal_date"] = today_str
    save_state(state)
    log.info("Daily signal pass complete.")


MARKET_CLOSE = dtime(15, 30)  # process self-stops here; Python Strategy Host's own
                               # schedule_stop (15:25) normally kills it first

if __name__ == "__main__":
    # The Python Strategy Host starts this as ONE long-lived process at
    # schedule_start and kills it at schedule_stop -- it does NOT re-invoke the
    # script every 15 minutes itself (confirmed against the intraday sibling's
    # own deployed copy, which uses this identical while-True-plus-sleep
    # pattern). main() dispatches internally (see its own docstring): the
    # first call each day runs the full daily pass, every subsequent call
    # runs only the intraday stop-check safety net.
    while True:
        try:
            main()
        except Exception:
            log.exception("Unhandled error in main()")
        if datetime.now(IST).time() >= MARKET_CLOSE:
            log.info("Market close reached — stopping")
            break
        time.sleep(900)  # 15 minutes
