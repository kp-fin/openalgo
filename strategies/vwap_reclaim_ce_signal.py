"""
VWAP Reclaim CE — OpenAlgo Forward Test
Run every 15 minutes: 9:15–13:05 IST (Mon–Fri)

Signal: Nifty reclaims anchored VWAP on 15m
  Daily EMA100 filter: spot > EMA100 (bull days only)
  Stretch: VWAP − low of day ≥ 20 pts (price stretched below VWAP)
  Reclaim: close > VWAP (price returns above VWAP)
  → BUY ATM CE at next candle open

Stop: reclaim candle low (spot)
Target: entry + 1.5 × stretch (1.5 × VWAP stretch)
Hard exit: 14:30 IST | New entry cutoff: 13:00 IST
State: state/vwap_reclaim_ce_state.json
"""

import json
import logging
import os
from datetime import date, datetime, time as dtime, timedelta

import pandas as pd
import pytz
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("vwap_reclaim_ce")

API_KEY      = os.getenv("OPENALGO_API_KEY", "your_openalgo_api_key_here")
HOST         = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
LOT_SIZE     = 75
ENTRY_END    = dtime(13, 0)
HARD_EXIT    = dtime(14, 30)
STRETCH_MIN  = 20        # pts below VWAP required before reclaim
RISK_MULT    = 1.5
EMA_PERIOD   = 100       # daily EMA for bull-day filter
IST          = pytz.timezone("Asia/Kolkata")
STATE_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "vwap_reclaim_ce_state.json")

os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)


def load_state():
    today = str(date.today())
    if os.path.exists(STATE_FILE):
        try:
            s = json.load(open(STATE_FILE))
            if s.get("date") == today:
                return s
        except Exception:
            pass
    return {"date": today, "ema100": None, "traded": False, "position": None,
            "below_vwap_seen": False}

def save_state(s):
    json.dump(s, open(STATE_FILE, "w"), default=str)

def _h():
    return {"Content-Type": "application/json"}

def get_candles(interval, start, end):
    r = requests.post(f"{HOST}/api/v1/history",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NSE_INDEX",
                            "interval": interval, "start_date": start, "end_date": end},
                      headers=_h(), timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(data)
    df = pd.DataFrame(data["data"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index("datetime").sort_index()

def get_ltp(symbol, exchange="NFO"):
    r = requests.post(f"{HOST}/api/v1/quotes",
                      json={"apikey": API_KEY, "symbol": symbol, "exchange": exchange},
                      headers=_h(), timeout=10)
    r.raise_for_status()
    return float(r.json()["data"]["ltp"])

def get_nifty_spot():
    r = requests.post(f"{HOST}/api/v1/quotes",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NSE_INDEX"},
                      headers=_h(), timeout=10)
    r.raise_for_status()
    return float(r.json()["data"]["ltp"])

def get_atm_ce_symbol(spot):
    strike = round(spot / 50) * 50
    r = requests.post(f"{HOST}/api/v1/optionsymbol",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NFO",
                            "expiry": "current", "strike": strike, "optiontype": "CE"},
                      headers=_h(), timeout=10)
    r.raise_for_status()
    return r.json()["data"]["symbol"], strike

def place_order(symbol, action):
    r = requests.post(f"{HOST}/api/v1/placeorder",
                      json={"apikey": API_KEY, "symbol": symbol, "exchange": "NFO",
                            "action": action, "quantity": LOT_SIZE,
                            "price_type": "MARKET", "product": "MIS"},
                      headers=_h(), timeout=15)
    r.raise_for_status()
    return r.json()

def compute_daily_ema100():
    """EMA100 on daily closes (last 200 trading days for warm-up)."""
    end   = date.today().isoformat()
    start = (date.today() - timedelta(days=400)).isoformat()
    df    = get_candles("1d", start, end)
    if len(df) < EMA_PERIOD:
        raise RuntimeError("Insufficient daily data for EMA100")
    return float(df["close"].ewm(span=EMA_PERIOD, adjust=False).mean().iloc[-1])

def compute_anchored_vwap(df_intraday):
    """Session-anchored VWAP from first bar of today."""
    typical = (df_intraday["high"] + df_intraday["low"] + df_intraday["close"]) / 3
    vol     = df_intraday["volume"]
    vwap    = (typical * vol).cumsum() / vol.cumsum()
    return vwap


def main():
    now = datetime.now(IST)
    t   = now.time()
    log.info(f"VWAP Reclaim CE check — {now.strftime('%H:%M:%S IST')}")

    state = load_state()

    # EMA100 computed once per day
    if state["ema100"] is None:
        try:
            state["ema100"] = compute_daily_ema100()
            log.info(f"EMA100 = {state['ema100']:.2f}")
            save_state(state)
        except Exception as e:
            log.warning(f"EMA100 compute failed: {e}")
            return

    ema100 = state["ema100"]

    # Exit management
    if state["position"]:
        pos = state["position"]
        try:
            ltp        = get_ltp(pos["symbol"])
            entry_ltp  = pos["entry_price"]
            target_ltp = pos["target_ltp"]
            stop_ltp   = pos["stop_ltp"]

            if ltp >= target_ltp:
                place_order(pos["symbol"], "SELL")
                log.info(f"EXIT TARGET {pos['symbol']} entry={entry_ltp:.2f} ltp={ltp:.2f}")
                state["position"] = None
            elif ltp <= stop_ltp:
                place_order(pos["symbol"], "SELL")
                log.info(f"EXIT STOP {pos['symbol']} entry={entry_ltp:.2f} ltp={ltp:.2f}")
                state["position"] = None
            elif t >= HARD_EXIT:
                place_order(pos["symbol"], "SELL")
                log.info(f"EXIT HARD {pos['symbol']} entry={entry_ltp:.2f} ltp={ltp:.2f}")
                state["position"] = None
            else:
                pct = (ltp - entry_ltp) / entry_ltp
                log.info(f"HOLD {pos['symbol']} entry={entry_ltp:.2f} ltp={ltp:.2f} ({pct*100:+.1f}%)")
        except Exception as e:
            log.warning(f"Exit check failed: {e}")
        save_state(state)

    if t > ENTRY_END or t >= HARD_EXIT or state["traded"]:
        return

    # Entry check
    try:
        today = date.today().isoformat()
        df    = get_candles("15m", today, today)
        if df.empty:
            return

        spot = get_nifty_spot()

        # Bull-day filter
        if spot < ema100:
            log.info(f"Bear day (spot {spot:.2f} < EMA100 {ema100:.2f}) — skip")
            return

        vwap = compute_anchored_vwap(df)
        vwap_now     = float(vwap.iloc[-1])
        last_close   = float(df["close"].iloc[-1])
        last_low     = float(df["low"].iloc[-1])
        day_low      = float(df["low"].min())

        stretch = vwap_now - day_low

        # Track if we've seen price below VWAP this session
        below_vwap_bars = (df["close"] < vwap).any()
        if below_vwap_bars:
            state["below_vwap_seen"] = True
            save_state(state)

        if not state["below_vwap_seen"]:
            log.info(f"No stretch seen yet (price hasn't been below VWAP) — wait")
            return

        if stretch < STRETCH_MIN:
            log.info(f"Stretch {stretch:.1f} < {STRETCH_MIN} — not enough stretch")
            return

        # Reclaim: current close > VWAP
        if last_close <= vwap_now:
            log.info(f"No reclaim: close={last_close:.2f} vwap={vwap_now:.2f}")
            return

        # Signal confirmed
        sym, strike = get_atm_ce_symbol(spot)
        entry_ltp   = get_ltp(sym)

        # Stop = reclaim candle low (spot); target = 1.5× stretch above entry
        stop_pts    = last_low
        risk_pts    = spot - stop_pts
        target_spot = spot + RISK_MULT * stretch

        # Convert to option premium pct proxy
        risk_pct_spot   = risk_pts / spot
        target_pct_prem = min(0.60, RISK_MULT * stretch / spot)
        stop_pct_prem   = min(0.40, risk_pct_spot)

        place_order(sym, "BUY")
        log.info(f"ENTRY VWAP Reclaim | {sym} | spot={spot:.2f} vwap={vwap_now:.2f} stretch={stretch:.1f} LTP={entry_ltp:.2f}")

        state["traded"]   = True
        state["position"] = {
            "symbol":       sym,
            "strike":       strike,
            "entry_price":  entry_ltp,
            "entry_time":   now.isoformat(),
            "target_ltp":   round(entry_ltp * (1 + target_pct_prem), 2),
            "stop_ltp":     round(entry_ltp * (1 - stop_pct_prem), 2),
        }

    except Exception as e:
        log.error(f"Entry check failed: {e}")

    save_state(state)


if __name__ == "__main__":
    main()
