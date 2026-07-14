"""
BB Squeeze PE — OpenAlgo Forward Test
Run every 15 minutes: 9:15–15:05 IST (Mon–Fri)

Signal: Bollinger Band squeeze on Nifty 15m
  BB(20, 2σ) squeeze ≥ 3 consecutive candles (BB width ≤ threshold)
  ADX ≥ 20 AND −DI > +DI (bearish momentum)
  Volume > 1.1× 20-period avg
  → BUY ATM PE at next candle open (market)

Target: +35% premium | Stop: −25% premium | Hard exit: 15:00 IST
State: state/bb_squeeze_pe_state.json
"""

import json
import logging
import os
import time
from datetime import date, datetime, time as dtime

import numpy as np
import pandas as pd
import pytz
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bb_squeeze_pe")

API_KEY    = os.getenv("OPENALGO_API_KEY", "your_openalgo_api_key_here")
HOST       = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
LOT_SIZE   = 75
TARGET_PCT = 0.35
STOP_PCT   = 0.25
HARD_EXIT  = dtime(15, 0)
IST        = pytz.timezone("Asia/Kolkata")
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "bb_squeeze_pe_state.json")

BB_PERIOD    = 20
BB_STD       = 2.0
SQUEEZE_BARS = 3       # consecutive squeeze candles needed
ADX_PERIOD   = 14
ADX_MIN      = 20
VOL_MULT     = 1.1

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
    return {"date": today, "traded": False, "position": None}

def save_state(s):
    json.dump(s, open(STATE_FILE, "w"), default=str)

def _h():
    return {"Content-Type": "application/json"}

def get_candles_today(interval="15m"):
    today = date.today().isoformat()
    r = requests.post(f"{HOST}/api/v1/history",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NSE_INDEX",
                            "interval": interval, "start_date": today, "end_date": today},
                      headers=_h(), timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(data)
    df = pd.DataFrame(data["data"])
    df["datetime"] = pd.to_datetime(df["timestamp"])
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

def get_atm_pe_symbol(spot):
    strike = round(spot / 50) * 50
    r = requests.post(f"{HOST}/api/v1/optionsymbol",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NFO",
                            "expiry": "current", "strike": strike, "optiontype": "PE"},
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


def compute_bb_squeeze(df):
    """True if last SQUEEZE_BARS candles were all in squeeze (narrow BB)."""
    close = df["close"]
    mid   = close.rolling(BB_PERIOD).mean()
    std   = close.rolling(BB_PERIOD).std()
    upper = mid + BB_STD * std
    lower = mid - BB_STD * std
    width = upper - lower

    # Dynamic squeeze threshold: width < median(width, 20 bars)
    threshold = width.rolling(BB_PERIOD).median()
    in_squeeze = width < threshold

    if len(in_squeeze) < SQUEEZE_BARS:
        return False
    return bool(in_squeeze.iloc[-SQUEEZE_BARS:].all())


def compute_adx(df, period=ADX_PERIOD):
    """Returns (adx, plus_di, minus_di) for last bar."""
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    n     = len(high)
    if n < period + 1:
        return 0.0, 0.0, 0.0

    tr   = np.zeros(n)
    pdm  = np.zeros(n)
    ndm  = np.zeros(n)

    for i in range(1, n):
        tr[i]  = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        up     = high[i] - high[i-1]
        down   = low[i-1] - low[i]
        pdm[i] = up   if (up > down and up > 0)   else 0
        ndm[i] = down if (down > up and down > 0) else 0

    def rma(arr, p):
        out = np.zeros(n)
        out[p] = arr[1:p+1].sum()
        for i in range(p+1, n):
            out[i] = out[i-1] - out[i-1]/p + arr[i]
        return out

    atr   = rma(tr, period)
    apdm  = rma(pdm, period)
    andm  = rma(ndm, period)

    eps = 1e-9
    pdi = 100 * apdm / (atr + eps)
    ndi = 100 * andm / (atr + eps)
    dx  = 100 * np.abs(pdi - ndi) / (pdi + ndi + eps)

    adx = rma(dx, period)
    return float(adx[-1]), float(pdi[-1]), float(ndi[-1])


def signal_detected(df):
    if len(df) < BB_PERIOD + SQUEEZE_BARS:
        return False
    if not compute_bb_squeeze(df):
        return False

    adx, pdi, ndi = compute_adx(df)
    if adx < ADX_MIN:
        return False
    if ndi <= pdi:
        return False

    vol    = df["volume"].iloc[-1]
    avg_vol = df["volume"].rolling(BB_PERIOD).mean().iloc[-1]
    if vol < VOL_MULT * avg_vol:
        return False

    log.info(f"Signal confirmed: ADX={adx:.1f} -DI={ndi:.1f} +DI={pdi:.1f} vol={vol:.0f} avg={avg_vol:.0f}")
    return True


def main():
    now = datetime.now(IST)
    t   = now.time()
    log.info(f"BB Squeeze PE check — {now.strftime('%H:%M:%S IST')}")

    state = load_state()

    # Exit open position
    if state["position"]:
        pos = state["position"]
        try:
            ltp   = get_ltp(pos["symbol"])
            entry = pos["entry_price"]
            pct   = (ltp - entry) / entry

            if pct >= TARGET_PCT:
                place_order(pos["symbol"], "SELL")
                log.info(f"EXIT TARGET {pos['symbol']} entry={entry:.2f} ltp={ltp:.2f} (+{pct*100:.1f}%)")
                state["position"] = None
            elif pct <= -STOP_PCT:
                place_order(pos["symbol"], "SELL")
                log.info(f"EXIT STOP {pos['symbol']} entry={entry:.2f} ltp={ltp:.2f} ({pct*100:.1f}%)")
                state["position"] = None
            elif t >= HARD_EXIT:
                place_order(pos["symbol"], "SELL")
                log.info(f"EXIT HARD {pos['symbol']} entry={entry:.2f} ltp={ltp:.2f} ({pct*100:.1f}%)")
                state["position"] = None
            else:
                log.info(f"HOLD {pos['symbol']} entry={entry:.2f} ltp={ltp:.2f} ({pct*100:+.1f}%)")
        except Exception as e:
            log.warning(f"Exit check failed: {e}")
        save_state(state)

    if t >= HARD_EXIT or state["traded"]:
        return

    # Entry check
    try:
        df = get_candles_today("15m")
        if signal_detected(df):
            spot        = get_nifty_spot()
            sym, strike = get_atm_pe_symbol(spot)
            entry_ltp   = get_ltp(sym)
            place_order(sym, "BUY")
            log.info(f"ENTRY BB Squeeze PE | {sym} | spot={spot:.2f} | LTP={entry_ltp:.2f}")
            state["traded"]   = True
            state["position"] = {"symbol": sym, "strike": strike, "entry_price": entry_ltp,
                                  "entry_time": now.isoformat()}
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
