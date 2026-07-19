"""
EMA Regime Crossover — Swing — 20-name equity universe — OpenAlgo Forward Test
Separate strategy from the intraday EMA Regime Crossover (ema_regime_crossover_signal.py) --
not a modification of it. Multi-day/multi-week holding, not intraday: no hard exit, no daily
state reset. See equities-system/strategies/ema_regime_crossover_swing.md for the full backtest
and parameter-tuning trail (2026-07-19) this script implements.

Regime (daily bars, JdK RRG RS-Ratio/RS-Momentum) REPLACES EMA(200) entirely -- each stock is
compared against its own respective index (NIFTY for the Nifty 50 pool, NIFTYNXT50 for the
Nifty Next 50 pool):
  RS          = 100 * (stock_close / index_close)
  RS-Ratio    = smooth(100 + (RS - SMA(RS,10)) / STDEV(RS,10), 2)          -- outperformance
  RS-Momentum = smooth(100 + (RS-Ratio - SMA(RS-Ratio,10)) / STDEV(RS-Ratio,10), 2)
  BULL regime: RS-Ratio > 100 AND (RS-Momentum > 100 OR RS-Momentum risen over the last
               RS_MOMENTUM_SLOPE_LOOKBACK days) -- "accelerating, or moving towards acceleration"

Entry (daily, EMA9/EMA20 crossover): bullish crossover while in BULL regime -> LONG only (the
long+short backtest showed shorts were a net drag -- dropped 2026-07-19, see the spec). Entry at
current LTP during the once-daily signal pass (approximates "next day's open", matching the
backtest's entry-at-next-open convention, since this script runs once per day near market open).

Exit: initial stop = entry - 3.25xATR(14, daily), then a CHANDELIER-STYLE TRAILING STOP
(6.0xATR(14, daily), ATR recomputed daily, ratchets up only) instead of a fixed target, or an
opposite EMA9/EMA20 crossover (bear_cross), whichever comes first. NO time-based/hard exit -- a
position can run for days to weeks. The trailing stop level is fixed for the whole trading day
(only updated once, during the daily signal pass) but checked against LIVE LTP every poll, so a
breach is caught intraday rather than waiting for next day's bar.

Instrument: cash equity, MTF (Margin Trading Facility) -- Karan's call 2026-07-19, specifically
built (OpenAlgo platform-wide MTF support, same day) because MIS's same-day square-off is wrong
for a multi-week hold and plain CNC gives no leverage. Real Dhan MTF leverage varies per stock
(surveyed 2026-07-19 across Nifty 100: 2.38x-4.55x, see strategies/backtests/mtf_leverage_survey.csv)
-- this script calls Dhan's REAL margin calculator fresh for each new entry (get_mtf_leverage()
below) rather than assuming a flat number or importing the sandbox engine's paper-trading survey
table (that's Flask-app-internal code; reaching into it from a standalone script would need a
second fragile cross-package sys.path hack for no real benefit over just calling the same real
API directly). Falls back to a conservative 3.0x if the live call fails for some reason.

Capital: allocated_capital starts at Rs 25,000 (Karan's confirmed figure, 2026-07-19), read fresh
from capital_state.py's capital_allocation.json each poll -- same shared mechanism as ORB_Spread
and the intraday EMA Regime Crossover. Live-mode closed trades compound their P&L into
allocated_capital; paper-mode trades never touch it (capital_state.record_trade() handles this).

Sizing: fixed 12.5% of (allocated_capital x per-symbol MTF leverage) per trade -- POSITION_PCT and
MAX_CONCURRENT (6) are INHERITED UNCHANGED from the intraday sibling's convention. The swing
strategy spec flags this as an open item (a ~40-day average hold ties up capital far longer per
position than intraday MIS turnover) -- not re-examined here, just carried forward as-is until
Karan revisits it.

Per-symbol position limit: implicit via the positions dict being keyed by symbol (no new entry
if one is already open in that name) -- same live-only divergence from the backtest as the
intraday sibling (the backtest's signal-generation pass doesn't prevent this, a real account
can't hold two independently-tracked entries in one name).

Universe: top 10 Nifty 50 + top 10 Nifty Next 50 by trailing 20-day average daily traded value,
re-ranked on the first poll of each new calendar month -- identical mechanism to the intraday
sibling's refresh_universe(), not re-derived here. A name dropping out of the monthly top-10
blocks new entries only; an existing open position rides to its own exit.

Scheduling: no hard exit. The script stops POLLING (not trading) at STOP_POLLING_TIME (15:30 IST)
each day so it doesn't run forever -- positions stay open in state, resumed when the Python
Strategy Host restarts this script fresh the next trading day. Poll interval 15 minutes (vs. the
intraday sibling's 5) -- daily-bar signals don't need tight polling; only the trailing-stop-vs-
live-LTP check benefits from periodic checking, and 15 min is frequent enough for that. The
DAILY SIGNAL PASS (regime/crossover computation, trailing-stop update, reverse-cross exit, new
entries) runs only once per day, gated on state["last_signal_date"] != today -- every other poll
that day only re-checks open positions' live LTP against their (unchanged-until-tomorrow) trailing
stop.

State persisted to state/ema_regime_crossover_swing_state.json -- positions and universe/month
BOTH persist across days (unlike the intraday sibling, which resets positions daily at hard exit).
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
log = logging.getLogger("ema_regime_crossover_swing")

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("OPENALGO_API_KEY", "your_openalgo_api_key_here")
HOST    = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
IST     = pytz.timezone("Asia/Kolkata")

RS_ZSCORE_WINDOW           = 10    # JdK published default
RS_SMOOTH_WINDOW           = 2     # JdK published default
RS_MOMENTUM_SLOPE_LOOKBACK = 2     # locked 2026-07-19 (best point of a 1-20 day sweep)

ENTRY_FAST = 9
ENTRY_SLOW = 20
ATR_PERIOD = 14

INITIAL_STOP_ATR_MULT = 3.25   # locked 2026-07-19
TRAIL_ATR_MULT        = 6.0    # locked 2026-07-19 (PF/total P&L plateau from here onward)

STOP_POLLING_TIME = dtime(15, 30)   # stop polling for the day, not a square-off -- see module docstring
POLL_SECONDS       = 15 * 60        # 15 minutes -- lighter than the intraday sibling's 5
DAILY_LOOKBACK_DAYS = 90            # comfortably covers RS/EMA/ATR warmup with buffer

POSITION_PCT   = 0.125    # inherited unchanged from the intraday sibling -- open item, see docstring
MAX_CONCURRENT = 6

UNIVERSE_EACH = 10

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

INDEX_FOR = {**{s: "NIFTY" for s in NIFTY_50}, **{s: "NIFTYNXT50" for s in NIFTY_NEXT_50}}

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "ema_regime_crossover_swing_state.json")
os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)


# ── State helpers ─────────────────────────────────────────────────────────────
def load_state():
    """Unlike the intraday sibling: positions and universe/month BOTH persist
    across days -- this is a multi-day-hold strategy, nothing resets daily."""
    if os.path.exists(STATE_FILE):
        try:
            s = json.load(open(STATE_FILE))
        except Exception:
            s = {}
    else:
        s = {}
    s.setdefault("positions", {})
    s.setdefault("universe", [])
    s.setdefault("universe_month", None)
    s.setdefault("last_signal_date", None)
    return s


def save_state(s):
    json.dump(s, open(STATE_FILE, "w"), default=str)


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
            log.warning(f"{symbol}: daily history fetch failed: {data}")
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
        log.warning(f"{symbol}: daily history fetch error: {e}")
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


def get_mtf_leverage(symbol, ltp):
    """Fresh Dhan margin-calculator call, product=MTF, 1-share BUY at current LTP --
    real per-symbol leverage, not a flat assumption (see module docstring)."""
    try:
        r = requests.post(f"{HOST}/api/v1/margin",
                          json={"apikey": API_KEY, "positions": [{
                              "exchange": "NSE", "symbol": symbol, "action": "BUY",
                              "quantity": "1", "product": "MTF", "pricetype": "MARKET",
                              "price": str(ltp),
                          }]}, headers=_headers(), timeout=10)
        resp = r.json()
        if resp.get("status") == "success":
            margin = float(resp["data"]["total_margin_required"])
            if margin > 0:
                return ltp / margin
        log.warning(f"{symbol}: margin call returned no usable margin, using fallback leverage: {resp}")
    except Exception as e:
        log.warning(f"{symbol}: live margin lookup failed, using conservative fallback: {e}")
    return 3.0  # conservative fallback -- below the survey's 2.38x-4.55x range's midpoint


def place_order(symbol, action, quantity):
    r = requests.post(f"{HOST}/api/v1/placeorder",
                      json={"apikey": API_KEY, "strategy": "ema_regime_crossover_swing", "symbol": symbol,
                            "exchange": "NSE", "action": action, "quantity": quantity,
                            "pricetype": "MARKET", "product": "MTF"},
                      headers=_headers(), timeout=15)
    r.raise_for_status()
    return r.json()


def log_closed_trade(sym, pos, exit_px, pnl, reason, now):
    """Append to the trade log (paper AND live) for ongoing Sharpe/PF/win-rate/
    avg-P&L tracking. Live mode also compounds pnl into allocated_capital --
    see capital_state.py."""
    record_trade("ema_regime_crossover_swing", {
        "date": now.strftime("%Y-%m-%d"),
        "symbol": sym,
        "direction": "LONG",
        "entry_time": pos["entry_time"],
        "entry_price": round(pos["entry_price"], 2),
        "exit_time": now.isoformat(),
        "exit_price": round(exit_px, 2),
        "qty": pos["qty"],
        "pnl_rupees": round(pnl, 2),
        "reason": reason,
    }, pnl)


def avg_daily_traded_value(symbol, lookback_days=20):
    df = get_candles_daily(symbol)
    if df is None or df.empty or "volume" not in df.columns:
        return None
    df = df.tail(lookback_days)
    if df.empty:
        return None
    return float((df["close"] * df["volume"]).mean())


def refresh_universe():
    """Rank both candidate pools by trailing 20-day ADV, take top 10 each --
    identical mechanism to the intraday sibling's refresh_universe(), not
    re-derived here. Only the RANKING is recomputed monthly; the candidate
    pools (NIFTY_50 / NIFTY_NEXT_50 above) are a static snapshot."""
    log.info("Universe refresh starting (monthly) ...")
    ranked = {}
    for pool_name, pool in (("Nifty 50", NIFTY_50), ("Nifty Next 50", NIFTY_NEXT_50)):
        rows = []
        for sym in pool:
            adv = avg_daily_traded_value(sym)
            if adv is not None:
                rows.append((sym, adv))
            time.sleep(0.1)
        rows.sort(key=lambda x: x[1], reverse=True)
        top = [sym for sym, _ in rows[:UNIVERSE_EACH]]
        ranked[pool_name] = top
        log.info(f"{pool_name} top {UNIVERSE_EACH}: {top}")
    universe = ranked["Nifty 50"] + ranked["Nifty Next 50"]
    log.info(f"Universe refreshed: {len(universe)} names")
    return universe


# ── Indicator helpers ────────────────────────────────────────────────────────
def compute_atr(df, period=ATR_PERIOD):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_rs_ratio_momentum(stock_close, index_close):
    """JdK RRG-style RS-Ratio / RS-Momentum, aligned on shared trading dates."""
    rs = 100 * (stock_close / index_close)
    rs_ratio_raw = 100 + (rs - rs.rolling(RS_ZSCORE_WINDOW).mean()) / rs.rolling(RS_ZSCORE_WINDOW).std()
    rs_ratio = rs_ratio_raw.rolling(RS_SMOOTH_WINDOW).mean()
    mom_raw = 100 + (rs_ratio - rs_ratio.rolling(RS_ZSCORE_WINDOW).mean()) / rs_ratio.rolling(RS_ZSCORE_WINDOW).std()
    rs_momentum = mom_raw.rolling(RS_SMOOTH_WINDOW).mean()
    return rs_ratio, rs_momentum


def compute_indicators(symbol, index_data_cache):
    """Daily OHLC + EMA9/20 cross + ATR14 + JdK RRG BULL regime. Returns None
    if there's not enough history for a valid reading yet."""
    df = get_candles_daily(symbol)
    if df.empty or len(df) < ENTRY_SLOW + RS_ZSCORE_WINDOW * 2 + 5:
        return None

    idx_symbol = INDEX_FOR.get(symbol)
    if idx_symbol is None:
        return None
    idx_df = index_data_cache.get(idx_symbol)
    if idx_df is None or idx_df.empty:
        return None

    combined_close = pd.DataFrame({"stock": df["close"], "index": idx_df["close"]}).dropna()
    rs_ratio, rs_momentum = compute_rs_ratio_momentum(combined_close["stock"], combined_close["index"])

    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=ENTRY_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ENTRY_SLOW, adjust=False).mean()
    df["atr"] = compute_atr(df)
    df["bull_cross"] = (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)) & (df["ema_fast"] > df["ema_slow"])
    df["bear_cross"] = (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)) & (df["ema_fast"] < df["ema_slow"])
    df["rs_ratio"] = rs_ratio.reindex(df.index)
    df["rs_momentum"] = rs_momentum.reindex(df.index)
    rs_momentum_rising = df["rs_momentum"] > df["rs_momentum"].shift(RS_MOMENTUM_SLOPE_LOOKBACK)
    bull_regime = (df["rs_ratio"] > 100) & ((df["rs_momentum"] > 100) | rs_momentum_rising)
    df["regime"] = np.where(bull_regime, "BULL", "NOT_BULL")

    df = df.dropna(subset=["atr", "rs_ratio", "rs_momentum"])
    return df if not df.empty else None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now(IST)
    t   = now.time()
    log.info(f"EMA Regime Crossover Swing check — {now.strftime('%H:%M:%S IST')}")

    if t >= STOP_POLLING_TIME:
        log.info("Past stop-polling time for today — positions stay open, resuming tomorrow.")
        return "stop"

    allocated_capital = get_allocated_capital("ema_regime_crossover_swing")
    state = load_state()

    # ── Monthly universe refresh (first poll of a new calendar month) ────────
    this_month = now.strftime("%Y-%m")
    if state["universe_month"] != this_month:
        try:
            state["universe"] = refresh_universe()
            state["universe_month"] = this_month
            save_state(state)
        except Exception as e:
            log.error(f"Universe refresh failed, keeping previous universe: {e}")
            if not state["universe"]:
                log.error("No universe available at all — cannot proceed this poll.")
                return

    universe = state["universe"]
    positions = state["positions"]

    # ── Once-per-day signal pass: regime/crossover, trailing-stop update,
    #    reverse-cross exits, new entries -- everything else is daily-cadence ──
    today_str = date.today().isoformat()
    if state.get("last_signal_date") != today_str:
        log.info("Running daily signal pass...")
        index_data = {}
        for idx_symbol in ("NIFTY", "NIFTYNXT50"):
            idf = get_candles_daily(idx_symbol, exchange="NSE_INDEX")
            if not idf.empty:
                index_data[idx_symbol] = idf
        if len(index_data) < 2:
            log.error("Could not fetch both index histories — skipping today's signal pass.")
        else:
            symbols_to_check = sorted(set(universe) | set(positions.keys()))
            indicators = {}
            for sym in symbols_to_check:
                ind = compute_indicators(sym, index_data)
                if ind is not None:
                    indicators[sym] = ind
                time.sleep(0.1)

            # -- Existing positions: reverse-cross exit, else trailing-stop update --
            for sym, pos in list(positions.items()):
                ind = indicators.get(sym)
                if ind is None:
                    continue
                last = ind.iloc[-1]
                if bool(last["bear_cross"]):
                    try:
                        ltp_map = get_multiquotes([sym])
                        exit_px = ltp_map.get(sym, pos["entry_price"])
                        place_order(sym, "SELL", pos["qty"])
                        pnl = (exit_px - pos["entry_price"]) * pos["qty"]
                        log.info(f"EXIT REVERSE_CROSS {sym} entry={pos['entry_price']:.2f} exit={exit_px:.2f} pnl={pnl:+.0f}")
                        log_closed_trade(sym, pos, exit_px, pnl, "REVERSE_CROSS", now)
                        del positions[sym]
                    except Exception as e:
                        log.error(f"Reverse-cross exit failed for {sym}: {e}")
                    continue

                atr = float(last["atr"])
                pos["highest_high"] = max(pos.get("highest_high", pos["entry_price"]), float(last["high"]))
                candidate_trail = pos["highest_high"] - TRAIL_ATR_MULT * atr
                pos["trail_stop"] = max(pos["trail_stop"], candidate_trail)

            # -- New entries: bull_cross + BULL regime, under the concurrency cap,
            #    no existing position in that symbol --
            for sym in universe:
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

                leverage = get_mtf_leverage(sym, entry_price)
                buying_power = allocated_capital * leverage
                qty = int((buying_power * POSITION_PCT) // entry_price)
                if qty <= 0:
                    continue

                initial_stop = entry_price - INITIAL_STOP_ATR_MULT * atr
                try:
                    place_order(sym, "BUY", qty)
                    positions[sym] = {
                        "qty": qty, "entry_price": entry_price, "entry_time": now.isoformat(),
                        "initial_stop": initial_stop, "trail_stop": initial_stop,
                        "highest_high": entry_price, "leverage": leverage,
                    }
                    log.info(f"ENTRY LONG {sym} qty={qty} entry={entry_price:.2f} "
                             f"stop={initial_stop:.2f} atr={atr:.2f} leverage={leverage:.2f}x")
                except Exception as e:
                    log.error(f"Entry failed for {sym}: {e}")

            state["last_signal_date"] = today_str

        save_state(state)

    # ── Every poll: check open positions' live LTP against the (daily-updated)
    #    trailing stop -- this is the responsive, intraday-checked part ───────
    if positions:
        try:
            ltp_map = get_multiquotes(list(positions.keys()))
        except Exception as e:
            log.warning(f"Position LTP fetch failed: {e}")
            ltp_map = {}

        for sym, pos in list(positions.items()):
            ltp = ltp_map.get(sym)
            if ltp is None:
                continue
            if ltp <= pos["trail_stop"]:
                try:
                    place_order(sym, "SELL", pos["qty"])
                    pnl = (ltp - pos["entry_price"]) * pos["qty"]
                    log.info(f"EXIT TRAIL_STOP {sym} entry={pos['entry_price']:.2f} exit={ltp:.2f} pnl={pnl:+.0f}")
                    log_closed_trade(sym, pos, ltp, pnl, "TRAIL_STOP", now)
                    del positions[sym]
                except Exception as e:
                    log.error(f"Trail-stop exit failed for {sym}: {e}")
            else:
                pnl_now = (ltp - pos["entry_price"]) * pos["qty"]
                log.info(f"HOLD LONG {sym} entry={pos['entry_price']:.2f} ltp={ltp:.2f} "
                         f"trail={pos['trail_stop']:.2f} pnl~={pnl_now:+.0f}")

        save_state(state)


if __name__ == "__main__":
    while True:
        try:
            if main() == "stop":
                break
        except Exception:
            log.exception("Unhandled error in main()")
        time.sleep(POLL_SECONDS)
