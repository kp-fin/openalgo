"""
PDH Breakout CE — OpenAlgo Forward Test
Run every 15 minutes: 9:15–11:20 IST (Mon–Fri)

Signal: Nifty closes above Previous Day High on 15m candle
  close > PDH
  Volume > 1.3× 20-period avg
  Entry window: 9:15–11:15 IST
  → BUY ATM CE at next candle open

Stop: PDH − 10 pts (spot)
Target: entry + 1.5 × (entry − stop) (1.5R)
Hard exit: 14:30 IST
State: state/pdh_breakout_ce_state.json
"""

import json
import logging
import os
import time
from datetime import date, datetime, time as dtime, timedelta

import pandas as pd
import pytz
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pdh_breakout_ce")

API_KEY    = os.getenv("OPENALGO_API_KEY", "your_openalgo_api_key_here")
HOST       = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
LOT_SIZE   = 65
ENTRY_END  = dtime(11, 15)
HARD_EXIT  = dtime(14, 30)
VOL_MULT   = 1.3
VOL_PERIOD = 20
PDH_BUFFER = 10          # stop = PDH − 10
RISK_MULT  = 1.5         # target = entry + 1.5 × risk
IST        = pytz.timezone("Asia/Kolkata")
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "pdh_breakout_ce_state.json")

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
    return {"date": today, "pdh": None, "traded": False, "position": None}

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
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return df.set_index("datetime").sort_index()

def get_pdh():
    """Previous trading day high from daily candles."""
    today = date.today()
    start = (today - timedelta(days=7)).isoformat()   # enough buffer for weekends
    end   = (today - timedelta(days=1)).isoformat()
    df    = get_candles("D", start, end)
    if df.empty:
        raise RuntimeError("No daily data for PDH")
    return float(df["high"].iloc[-1])

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

def get_current_expiry():
    r = requests.post(f"{HOST}/api/v1/expiry",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NFO", "instrumenttype": "options"},
                      headers=_h(), timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success" or not data.get("data"):
        raise RuntimeError(data)
    return data["data"][0].replace("-", "")

def get_atm_ce_symbol(spot):
    strike = round(spot / 50) * 50
    expiry = get_current_expiry()
    r = requests.post(f"{HOST}/api/v1/optionsymbol",
                      json={"apikey": API_KEY, "underlying": "NIFTY", "exchange": "NSE_INDEX",
                            "expiry_date": expiry, "offset": "ATM", "option_type": "CE"},
                      headers=_h(), timeout=10)
    r.raise_for_status()
    return r.json()["symbol"], strike

def place_order(symbol, action):
    r = requests.post(f"{HOST}/api/v1/placeorder",
                      json={"apikey": API_KEY, "strategy": "pdh_breakout_ce", "symbol": symbol, "exchange": "NFO",
                            "action": action, "quantity": LOT_SIZE,
                            "pricetype": "MARKET", "product": "MIS"},
                      headers=_h(), timeout=15)
    r.raise_for_status()
    return r.json()


def main():
    now = datetime.now(IST)
    t   = now.time()
    log.info(f"PDH Breakout CE check — {now.strftime('%H:%M:%S IST')}")

    state = load_state()

    # Fetch PDH once per day
    if state["pdh"] is None:
        try:
            state["pdh"] = get_pdh()
            log.info(f"PDH = {state['pdh']:.2f}")
            save_state(state)
        except Exception as e:
            log.warning(f"PDH fetch failed: {e}")
            return

    pdh = state["pdh"]

    # Exit management (premium-based)
    if state["position"]:
        pos = state["position"]
        try:
            ltp        = get_ltp(pos["symbol"])
            entry_ltp  = pos["entry_price"]
            target_ltp = entry_ltp * (1 + pos["target_pct"])
            stop_ltp   = entry_ltp * (1 - pos["stop_pct"])

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

    # Entry: close above PDH with volume
    try:
        today = date.today().isoformat()
        df    = get_candles("15m", today, today)
        if df.empty or len(df) < VOL_PERIOD:
            return

        last_close  = float(df["close"].iloc[-1])
        last_vol    = float(df["volume"].iloc[-1])
        avg_vol     = float(df["volume"].rolling(VOL_PERIOD).mean().iloc[-1])

        if last_close > pdh and last_vol > VOL_MULT * avg_vol:
            spot        = get_nifty_spot()
            sym, strike = get_atm_ce_symbol(spot)
            entry_ltp   = get_ltp(sym)

            # R-based risk on spot, applied as premium pct proxy
            stop_pts    = pdh - PDH_BUFFER
            risk_pts    = spot - stop_pts
            target_pts  = risk_pts * RISK_MULT
            # convert to premium pct (rough proxy)
            target_pct  = min(0.50, target_pts / spot)
            stop_pct    = min(0.30, (spot - stop_pts) / spot)

            place_order(sym, "BUY")
            log.info(f"ENTRY PDH Breakout | {sym} | spot={spot:.2f} pdh={pdh:.2f} LTP={entry_ltp:.2f}")
            state["traded"]   = True
            state["position"] = {"symbol": sym, "strike": strike, "entry_price": entry_ltp,
                                  "entry_time": now.isoformat(),
                                  "target_pct": target_pct, "stop_pct": stop_pct}
        else:
            log.info(f"No signal: close={last_close:.2f} PDH={pdh:.2f} vol={last_vol:.0f} avg={avg_vol:.0f}")
    except Exception as e:
        log.error(f"Entry check failed: {e}")

    save_state(state)


if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception:
            log.exception("Unhandled error in main()")
        if datetime.now(IST).time() >= HARD_EXIT:
            log.info("Hard exit reached — stopping")
            break
        time.sleep(900)  # 15 minutes
