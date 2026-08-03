"""
ORB_Spread — Peachy Rejection Method, Debit Spread Execution — OpenAlgo Forward Test
(formerly ORB v2, then ORB_kp — renamed ORB_Spread 2026-07-16)
Run every 5 minutes: 9:45-15:15 IST (Mon-Fri)

Opening Range: 9:15-9:45 IST (30m)
Bearish Reject -> Bear Put Spread  (BUY ATM PE + SELL OTM1 PE)
Bullish Reject -> Bull Call Spread (BUY ATM CE + SELL OTM1 CE)
Lower High     -> Bear Put Spread  (confirmation, if no primary)
Higher Low     -> Bull Call Spread (confirmation, if no primary)
Range day skip: price inside OR at 10:15 -> no new entries today
Previous-day filter (direction-aware, changed 2026-08-04): if yesterday's
net directional move exceeds PREV_MOVE_THRESHOLD, BULL (CE) entries are
blocked for the day but BEAR (PE) entries are still allowed. Added
2026-07-16 as a blanket block on both directions (backtest: quiet prior
day PF 1.38 vs big-trend prior day PF 0.98); direction split tested
2026-07-23 (big-trend SHORT PF 1.16 vs LONG PF 0.84 -- LONG is the weak
side, not SHORT) and re-verified 2026-08-03 against the adopted 50pt/15%
spread model (candidate rule spread PF 4.50 vs current-live 4.64, but
+45% more total spread points, 8,180 vs 5,654 -- see orb_spread.md,
"Previous-Day Net Move" and "Direction-Aware Big-Trend Filter" sections).
Karan's call 2026-08-04: adopt the direction-aware version, accepting the
modest PF dilution for the absolute P&L gain. Fails OPEN (does not block
trading) if the fetch fails -- this is an added filter on an
already-working strategy, not a risk control, so a fetch outage should
not blank the whole day.

Exit decision is based on NIFTY SPOT movement, not spread premium percent
— matches the backtest methodology exactly (indices-system/strategies/orb_spread.md,
"Debit Spread Structure" section):
    Target: +40 pts spot move in favor | Stop: -25 pts spot move against
    Hard exit: 15:15 IST (changed from 14:30, 2026-07-16 -- see "Hard Exit Time" test)

Spread width: 50 pts (ATM long leg + OTM1 short leg = one real NIFTY
strike interval). Adopted from backtest evidence — a narrow/cheap spread
caps max loss below the strategy's own -25pt stop while barely capping
the +40pt target, producing backtest PF 3.99 (50pt/15%-assumed-cost) vs
naked-long PF 1.14 and a 100pt/35%-cost spread's PF 0.30. Real cost is
whatever the market actually charges at entry, not an assumption — this
script tracks it directly from live leg prices.

Execution (reworked 2026-07-22): BOTH entry and exit place two plain
/api/v1/placeorder calls against exact, pre-resolved symbols. Entry
trades the same symbols that were resolved and quoted during sizing
(estimate_spread_capital_requirement) — previously entry re-resolved
ATM/OTM1 offsets via /api/v1/optionsmultiorder moments after sizing,
and spot ticking across a strike boundary in between could open a
different strike than the one sized. Leg ordering is live-safe: entry
buys the long leg before selling the short (hedge exists first); exit
buys the short back before selling the long (never leaves the short
momentarily naked). Exit records per-leg completion in state so a
partial failure retried next tick never re-sends an already-closed leg.
Exit still deliberately never re-resolves offsets — by exit time spot
has moved (that's why we're exiting), so "ATM" could map to a different
strike than what we actually hold.

State persisted to state/orb_spread_state.json (intraday)
"""

import json
import logging
import os
import sys
import time
from datetime import date, datetime, time as dtime, timedelta

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
log = logging.getLogger("orb_spread")

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY       = os.getenv("OPENALGO_API_KEY", "your_openalgo_api_key_here")
HOST          = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
LOT_SIZE      = 65
OR_MIN        = 30
OR_MAX        = 150
TARGET_PTS    = 40          # NIFTY spot points (matches backtest, not premium %)
STOP_PTS      = 25
SHORT_OFFSET  = "OTM1"      # short leg = one real strike interval from ATM (~50pts)
PREV_MOVE_THRESHOLD = 0.42  # % -- matches the backtest's entry-level median split (2026-07-16)
ENTRY_END     = dtime(12, 0)
HARD_EXIT     = dtime(15, 15)   # changed from 14:30, 2026-07-16 -- see orb_spread.md "Hard Exit Time" test
RANGE_CHK     = dtime(10, 15)
IST           = pytz.timezone("Asia/Kolkata")
STATE_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "orb_spread_state.json")

# Debit-spread payoff model constants (mirrors the adopted backtest config --
# orb_spread.md "Debit Spread Structure", 50pt/15%-cost). REFERENCE ONLY as of
# 2026-07-22: no longer used in any live calculation. P&L logging uses real
# fills (2026-07-21 fix), and sizing uses the real /api/v1/margin basket quote
# floored at the real net debit (2026-07-22 fix). Kept to document the model
# the backtest evidence rests on.
SPREAD_WIDTH  = 50
SPREAD_COST   = SPREAD_WIDTH * 0.15  # 7.5 pts

# Capital-based sizing (2026-07-18, redesigned 2026-07-21 three times, Karan-
# confirmed): bull and bear each get their own FIXED, INDEPENDENT capital
# pool -- (allocated_capital x DEPLOYMENT_CAP_PCT) / 2 each -- rather than
# sharing one combined pool (that was tried first; replaced same day because
# whichever signal fired first could claim most/all of a shared pool,
# starving or zeroing the second same-day entry). Each leg's cap is a
# fraction of allocated_capital, not a frozen rupee figure, so both scale up
# automatically as profit compounds in live mode. See compute_lot_quantity().
DEPLOYMENT_CAP_PCT = 0.80  # max total capital deployed across BOTH positions combined (2 x 40% each), Karan-confirmed 2026-07-21

os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

# ── State helpers ─────────────────────────────────────────────────────────────
def load_state():
    today = str(date.today())
    if os.path.exists(STATE_FILE):
        try:
            s = json.load(open(STATE_FILE))
            if s.get("date") == today:
                return s
        except Exception:
            pass
    return {"date": today, "orb_high": None, "orb_low": None, "or_locked": False,
            "range_day": False, "bear_traded": False, "bull_traded": False,
            "bear_position": None, "bull_position": None, "prev_move_pct": None}

def save_state(s):
    json.dump(s, open(STATE_FILE, "w"), default=str)

# ── OpenAlgo helpers ──────────────────────────────────────────────────────────
def _headers():
    return {"Content-Type": "application/json"}

def get_candles(interval="5m"):
    today_str = date.today().isoformat()
    r = requests.post(f"{HOST}/api/v1/history",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NSE_INDEX",
                            "interval": interval, "start_date": today_str, "end_date": today_str},
                      headers=_headers(), timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(f"History API error: {data}")
    df = pd.DataFrame(data["data"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.set_index("datetime").sort_index()
    # COMPLETED BARS ONLY (fixed 2026-07-22): the API returns the currently
    # FORMING bar too, so signal logic reading the last row was evaluating a
    # candle mid-formation -- a rejection/confirmation pattern could appear
    # and then vanish before the bar closed ("repainting"), which the
    # backtest (completed bars only) never saw. Drop any bar whose window
    # hasn't fully elapsed yet.
    bar_minutes = int(interval.rstrip("m")) if interval.endswith("m") else None
    if bar_minutes:
        now_naive = datetime.now(IST).replace(tzinfo=None)
        df = df[df.index + pd.Timedelta(minutes=bar_minutes) <= now_naive]
    return df

def get_prev_day_move_pct():
    """Previous trading day's net directional move as % of that day's open
    (|close - open| / open * 100). Regime filter -- see orb_spread.md
    "Previous-Day Net Move" test for why."""
    today = date.today()
    start = (today - timedelta(days=10)).isoformat()   # buffer for weekends/holidays
    end   = (today - timedelta(days=1)).isoformat()
    r = requests.post(f"{HOST}/api/v1/history",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NSE_INDEX",
                            "interval": "D", "start_date": start, "end_date": end},
                      headers=_headers(), timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success" or not data.get("data"):
        raise RuntimeError(f"History API error: {data}")
    last = data["data"][-1]   # most recent daily bar = yesterday's session
    return abs(last["close"] - last["open"]) / last["open"] * 100

def get_nifty_spot():
    r = requests.post(f"{HOST}/api/v1/quotes",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NSE_INDEX"},
                      headers=_headers(), timeout=10)
    r.raise_for_status()
    return float(r.json()["data"]["ltp"])

def get_current_expiry():
    r = requests.post(f"{HOST}/api/v1/expiry",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NFO", "instrumenttype": "options"},
                      headers=_headers(), timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success" or not data.get("data"):
        raise RuntimeError(data)
    return data["data"][0].replace("-", "")

def get_multiquotes(symbols):
    """symbols: list of (symbol, exchange) tuples. Returns {symbol: ltp}."""
    r = requests.post(f"{HOST}/api/v1/multiquotes",
                      json={"apikey": API_KEY,
                            "symbols": [{"symbol": s, "exchange": e} for s, e in symbols]},
                      headers=_headers(), timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(data)
    return {res["symbol"]: float(res["data"]["ltp"]) for res in data["results"]}

def resolve_option_symbol(option_type, offset, expiry):
    r = requests.post(f"{HOST}/api/v1/optionsymbol",
                      json={"apikey": API_KEY, "underlying": "NIFTY", "exchange": "NSE_INDEX",
                            "expiry_date": expiry, "offset": offset, "option_type": option_type},
                      headers=_headers(), timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(data)
    return data["symbol"]

def estimate_spread_capital_requirement(option_type):
    """Resolve the ATM/OTM1 leg symbols, then fetch BOTH the live net debit
    (for entry_net_debit tracking/logging) and the REAL basket margin for 1
    lot via /api/v1/margin -- same pattern already used by the swing
    sibling's get_mtf_leverage(). No order is placed by this call.

    Fixed 2026-07-22: compute_lot_quantity() used to size purely against
    net_debit (the spread's defined max loss), on the assumption the broker
    nets the two legs into one defined-risk position. It doesn't -- Sandbox
    (and per Karan's understanding, real Dhan margin) blocks EACH LEG'S
    NOTIONAL INDEPENDENTLY, not netted. At the settings live 2026-07-22 that
    produced ~192 lots/12,480 shares per leg -- the real margin for a
    position that size is roughly 4x the Rs 2,00,000 per-leg pool, the same
    class of bug as the 2026-07-21 sizing bug (see scorecard.md), just
    re-surfaced by a different constraint the previous fix didn't check.
    Also returns the resolved leg symbols so the ORDER trades the exact
    contracts that were sized (fixed 2026-07-22 -- entry previously
    re-resolved ATM/OTM1 offsets via /optionsmultiorder moments after
    sizing; spot ticking across a strike boundary in between could open a
    different strike than the one sized/quoted).

    Returns (net_debit_per_share, margin_per_lot, long_sym, short_sym)."""
    expiry = get_current_expiry()
    long_sym = resolve_option_symbol(option_type, "ATM", expiry)
    short_sym = resolve_option_symbol(option_type, SHORT_OFFSET, expiry)
    quotes = get_multiquotes([(long_sym, "NFO"), (short_sym, "NFO")])
    net_debit = quotes[long_sym] - quotes[short_sym]

    r = requests.post(f"{HOST}/api/v1/margin",
                      json={"apikey": API_KEY, "positions": [
                          {"exchange": "NFO", "symbol": long_sym, "action": "BUY",
                           "quantity": str(LOT_SIZE), "product": "MIS", "pricetype": "MARKET"},
                          {"exchange": "NFO", "symbol": short_sym, "action": "SELL",
                           "quantity": str(LOT_SIZE), "product": "MIS", "pricetype": "MARKET"},
                      ]}, headers=_headers(), timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Margin API error: {data}")
    margin_per_lot = float(data["data"]["total_margin_required"])
    return net_debit, margin_per_lot, long_sym, short_sym

def compute_lot_quantity(net_debit_per_share, margin_per_lot):
    """Capital-based sizing (2026-07-18, redesigned 2026-07-21 three times +
    2026-07-22 margin fix, Karan-confirmed): bull and bear each get their own
    FIXED, INDEPENDENT capital pool -- (allocated_capital x
    DEPLOYMENT_CAP_PCT) / 2 -- rather than sharing one combined pool. A
    shared-pool design was tried first and replaced same day (2026-07-21):
    whichever signal fired first could claim most/all of the shared amount,
    starving or zeroing the second same-day entry. Fixing each leg's own
    pool removes that order-of-arrival risk, at the cost of some capital
    efficiency on solo-signal days (a lone entry is now capped at half the
    deployment ceiling, not the full amount -- accepted tradeoff). The
    per-leg cap is a fraction of allocated_capital, not a frozen rupee
    figure, so it scales up automatically as profit compounds in live mode.

    Sizes against the REAL basket margin for 1 lot (see
    estimate_spread_capital_requirement), not net_debit -- see that
    function's docstring for why net_debit alone undersizes the real
    capital cost. Floored at net_debit x LOT_SIZE (the defined-risk floor
    the backtest itself assumes) so a margin-API hiccup returning something
    too small can't size UP past what the model expects either.

    Can return 0 if even 1 lot doesn't fit the per-leg cap at the real
    margin/premium -- the caller must skip the trade rather than force a
    minimum entry. No separate MAX_LOTS backstop -- removed 2026-07-21
    (Karan-confirmed) since it would silently override the capital-based
    settings; the per-leg capital math above is the sole sizing limit."""
    allocated_capital = get_allocated_capital("orb_spread")
    per_leg_cap = (allocated_capital * DEPLOYMENT_CAP_PCT) / 2
    capital_per_lot = max(margin_per_lot, net_debit_per_share * LOT_SIZE)
    lots = int(per_leg_cap // capital_per_lot)
    return lots * LOT_SIZE

def place_order(symbol, action, quantity):
    r = requests.post(f"{HOST}/api/v1/placeorder",
                      json={"apikey": API_KEY, "strategy": "orb_spread", "symbol": symbol, "exchange": "NFO",
                            "action": action, "quantity": quantity,
                            "pricetype": "MARKET", "product": "MIS"},
                      headers=_headers(), timeout=15)
    r.raise_for_status()
    return r.json()

def open_spread_legs(long_symbol, short_symbol, quantity):
    """Open the debit spread on the EXACT symbols already resolved and sized
    (estimate_spread_capital_requirement) -- replaces the old
    /optionsmultiorder offset-based entry, which re-resolved ATM/OTM1
    moments after sizing and could open a different strike than the one
    sized if spot ticked across a strike boundary in between (fixed
    2026-07-22).

    Leg order matters live: BUY the long leg FIRST, then SELL the short --
    the hedge exists before the short is opened, so a real broker never
    sees a naked short (margin spike / outright rejection risk). If the
    short leg fails after the long filled, retry once, then close the long
    (compensating SELL) and abort the entry -- never leave a silent naked
    long that no exit logic knows the true shape of.

    Returns (long_ltp, short_ltp) -- post-order quotes, the entry-fill
    proxies used for entry_net_debit tracking."""
    place_order(long_symbol, "BUY", quantity)
    try:
        place_order(short_symbol, "SELL", quantity)
    except Exception as first_err:
        log.warning(f"Short leg SELL failed after long leg filled ({first_err}) — retrying once")
        try:
            place_order(short_symbol, "SELL", quantity)
        except Exception:
            log.error("Short leg SELL failed twice — closing the long leg to avoid a naked position")
            try:
                place_order(long_symbol, "SELL", quantity)
                log.error("Compensating close of the long leg succeeded — entry fully aborted")
            except Exception:
                log.critical(f"COMPENSATING CLOSE FAILED — naked long {long_symbol} x{quantity} "
                             f"may be live; MANUAL INTERVENTION NEEDED")
            raise
    quotes = get_multiquotes([(long_symbol, "NFO"), (short_symbol, "NFO")])
    return quotes[long_symbol], quotes[short_symbol]

def close_spread(pos):
    """Close both legs using their EXACT entry symbols -- NOT re-resolved
    offsets, since spot has moved since entry (that's why we're exiting)
    and re-resolving ATM/OTM1 now could target a different strike.

    Leg order flipped 2026-07-22: BUY back the short leg FIRST, then SELL
    the long -- closing the long first leaves the short momentarily naked,
    which a real broker can margin-reject mid-exit. Per-leg completion is
    recorded on the position dict (persisted via state) so a partial
    failure retried next tick does NOT re-send a leg that already closed --
    re-sending would flip or double the position instead of closing it.

    Returns (long_exit_ltp, short_exit_ltp) -- post-order quotes, so the
    caller can log REAL fill-based P&L (see log_closed_trade(), fixed
    2026-07-21) instead of a modeled estimate."""
    qty = pos["quantity"]
    if not pos.get("short_leg_closed"):
        place_order(pos["short_symbol"], "BUY", qty)
        pos["short_leg_closed"] = True
    if not pos.get("long_leg_closed"):
        place_order(pos["long_symbol"], "SELL", qty)
        pos["long_leg_closed"] = True
    quotes = get_multiquotes([(pos["long_symbol"], "NFO"), (pos["short_symbol"], "NFO")])
    return quotes[pos["long_symbol"]], quotes[pos["short_symbol"]]


def log_closed_trade(direction, reason, pos, spot, pnl_pts, exit_net_debit, now):
    """Real fill-based P&L (fixed 2026-07-21, replaces a modeled estimate).
    A debit spread's value = long_leg_premium - short_leg_premium: you PAY
    entry_net_debit to open (buy long, sell short) and RECEIVE exit_net_debit
    to close (sell long, buy short) -- P&L per share is exactly that
    difference, no modeling assumption needed. Old formula (spread_pnl_pts =
    spot points moved, capped at SPREAD_WIDTH, minus a flat SPREAD_COST
    assumption) could diverge substantially from real fills -- on 2026-07-21
    it logged +Rs 81,558.75/-Rs 17,062.50 for two trades whose real fills
    were +Rs 21,385.00/-Rs 14,673.75, a ~9.6x overstatement on the winner.
    See indices-system/scorecard.md for the full writeup and the restated
    historical CSV values. Live mode still compounds pnl_rupees into
    allocated_capital -- see capital_state.py."""
    qty = pos.get("quantity", LOT_SIZE)
    pnl_rupees = (exit_net_debit - pos["entry_net_debit"]) * qty
    record_trade("orb_spread", {
        "date": now.strftime("%Y-%m-%d"),
        "entry_time": pos["entry_time"],
        "direction": direction,
        "signal": pos["signal"],
        "entry_spot": round(pos["entry_spot"], 2),
        "exit_time": now.isoformat(),
        "exit_spot": round(spot, 2),
        "pnl_pts": round(pnl_pts, 2),
        "exit_net_debit": round(exit_net_debit, 2),
        "pnl_rupees": round(pnl_rupees, 2),
        "reason": reason,
    }, pnl_rupees)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now(IST)
    t   = now.time()
    log.info(f"ORB_Spread check — {now.strftime('%H:%M:%S IST')}")

    state = load_state()

    # ── Previous-day net move regime filter (computed once per day) ──────────
    # Retried every tick until it succeeds; never blocks the rest of main() --
    # fails OPEN (entries allowed) if the fetch is unavailable, since this is
    # an added filter on an already-working strategy, not a risk control.
    if state.get("prev_move_pct") is None:
        try:
            state["prev_move_pct"] = get_prev_day_move_pct()
            log.info(f"Prev-day net move = {state['prev_move_pct']:.2f}% "
                     f"(entries allowed only if <= {PREV_MOVE_THRESHOLD}%)")
            save_state(state)
        except Exception as e:
            log.warning(f"Prev-day move fetch failed (will retry next tick): {e}")

    # ── Build / update 30m OR ─────────────────────────────────────────────────
    if not state["or_locked"]:
        try:
            df = get_candles("5m")
            or_window = df.between_time("09:15", "09:44")
            if not or_window.empty:
                state["orb_high"] = float(or_window["high"].max())
                state["orb_low"]  = float(or_window["low"].min())

            if t >= dtime(9, 45) and state["orb_high"] is not None:
                or_range = state["orb_high"] - state["orb_low"]
                if or_range < OR_MIN or or_range > OR_MAX:
                    log.info(f"OR range {or_range:.0f} pts outside {OR_MIN}–{OR_MAX} — skipping day")
                    state["range_day"] = True
                state["or_locked"] = True
                log.info(f"OR locked: High={state['orb_high']:.2f}  Low={state['orb_low']:.2f}  Range={or_range:.0f}")
        except Exception as e:
            log.warning(f"OR build failed: {e}")
        save_state(state)
        if not state["or_locked"] or state["range_day"]:
            return

    orb_high = state["orb_high"]
    orb_low  = state["orb_low"]

    # ── Range day check at 10:15 ──────────────────────────────────────────────
    # No early return on detection (fixed 2026-07-22): this used to `return`
    # immediately, skipping the exit-management block below for one full tick
    # — a position opened between 09:45 and 10:15 went unmanaged on exactly
    # the tick that flagged the range day. The flag only needs to gate NEW
    # entries, and the "Range day — no new entries" gate below already does
    # that; exits must run every tick unconditionally.
    if t >= RANGE_CHK and not state.get("range_checked"):
        try:
            spot = get_nifty_spot()
            if orb_low < spot < orb_high:
                log.info(f"Range day detected at 10:15 (spot {spot:.2f} inside OR) — no more entries")
                state["range_day"] = True
            state["range_checked"] = True
            save_state(state)
        except Exception as e:
            log.warning(f"Range day check failed: {e}")

    # ── Exit open positions (spot-points based, matches backtest) ────────────
    for direction, sign in (("bear", 1), ("bull", -1)):
        pos = state.get(f"{direction}_position")
        if not pos:
            continue
        try:
            spot       = get_nifty_spot()
            entry_spot = pos["entry_spot"]
            pnl_pts    = (entry_spot - spot) * sign  # bear profits on spot falling, bull on spot rising

            if pnl_pts >= TARGET_PTS:
                long_exit_ltp, short_exit_ltp = close_spread(pos)
                exit_net_debit = long_exit_ltp - short_exit_ltp
                log.info(f"EXIT TARGET {direction.upper()} entry_spot={entry_spot:.2f} spot={spot:.2f} (+{pnl_pts:.1f}pts) "
                         f"exit_net_debit={exit_net_debit:.2f} (entry {pos['entry_net_debit']:.2f})")
                log_closed_trade(direction, "TARGET", pos, spot, pnl_pts, exit_net_debit, now)
                state[f"{direction}_position"] = None
            elif pnl_pts <= -STOP_PTS:
                long_exit_ltp, short_exit_ltp = close_spread(pos)
                exit_net_debit = long_exit_ltp - short_exit_ltp
                log.info(f"EXIT STOP {direction.upper()} entry_spot={entry_spot:.2f} spot={spot:.2f} ({pnl_pts:.1f}pts) "
                         f"exit_net_debit={exit_net_debit:.2f} (entry {pos['entry_net_debit']:.2f})")
                log_closed_trade(direction, "STOP", pos, spot, pnl_pts, exit_net_debit, now)
                state[f"{direction}_position"] = None
            elif t >= HARD_EXIT:
                long_exit_ltp, short_exit_ltp = close_spread(pos)
                exit_net_debit = long_exit_ltp - short_exit_ltp
                log.info(f"EXIT HARD {direction.upper()} entry_spot={entry_spot:.2f} spot={spot:.2f} ({pnl_pts:.1f}pts) "
                         f"exit_net_debit={exit_net_debit:.2f} (entry {pos['entry_net_debit']:.2f})")
                log_closed_trade(direction, "HARD_EXIT", pos, spot, pnl_pts, exit_net_debit, now)
                state[f"{direction}_position"] = None
            else:
                # Informational net-value check (not the exit decision)
                try:
                    quotes = get_multiquotes([(pos["long_symbol"], "NFO"), (pos["short_symbol"], "NFO")])
                    net_now = quotes[pos["long_symbol"]] - quotes[pos["short_symbol"]]
                    log.info(f"HOLD {direction.upper()} spot={spot:.2f} ({pnl_pts:+.1f}pts) "
                             f"net_debit_now={net_now:.2f} (entry {pos['entry_net_debit']:.2f})")
                except Exception:
                    log.info(f"HOLD {direction.upper()} spot={spot:.2f} ({pnl_pts:+.1f}pts)")
        except Exception as e:
            log.warning(f"Exit check failed for {direction}: {e}")

    save_state(state)

    # ── No new entries after 12:00, hard exit, or on a range day ─────────────
    if t > ENTRY_END or t >= HARD_EXIT:
        return
    if state["range_day"]:
        log.info("Range day — no new entries.")
        return
    big_trend_day = (state.get("prev_move_pct") is not None
                      and state["prev_move_pct"] > PREV_MOVE_THRESHOLD)
    if big_trend_day:
        log.info(f"Prev-day move {state['prev_move_pct']:.2f}% > {PREV_MOVE_THRESHOLD}% threshold — "
                  f"BULL entries blocked today, BEAR entries still allowed (direction-aware filter, 2026-08-04).")

    # ── Signal detection ──────────────────────────────────────────────────────
    try:
        df = get_candles("5m")
        sig = df.between_time("09:45", now.strftime("%H:%M"))
        if len(sig) < 2:
            return

        c  = sig["close"].values
        o  = sig["open"].values
        h  = sig["high"].values
        lo = sig["low"].values
        i  = len(sig) - 1

        bear_signal = bull_signal = None

        if c[i-1] > orb_high and c[i] < orb_high and c[i] < o[i]:
            bear_signal = "BearishReject"
        if c[i-1] < orb_low and c[i] > orb_low and c[i] > o[i]:
            bull_signal = "BullishReject"
        if bear_signal is None and i >= 2:
            if h[i] < h[i-1] < h[i-2] and c[i] < orb_high:
                bear_signal = "LowerHigh"
        if bull_signal is None and i >= 2:
            if lo[i] > lo[i-1] > lo[i-2] and c[i] > orb_low:
                bull_signal = "HigherLow"

        spot = float(c[i])

        # ── Enter Bear Put Spread ─────────────────────────────────────────────
        if bear_signal and not state["bear_traded"] and not state["bear_position"]:
            try:
                net_debit_est, margin_per_lot, long_sym, short_sym = estimate_spread_capital_requirement("PE")
                quantity = compute_lot_quantity(net_debit_est, margin_per_lot)
                if quantity <= 0:
                    log.info(f"BEAR signal {bear_signal} skipped — even 1 lot doesn't fit "
                             f"the per-leg capital pool (1-lot margin Rs {margin_per_lot:,.0f}, "
                             f"est net_debit {net_debit_est:.2f}). "
                             f"Not marking bear_traded -- will retry next tick.")
                else:
                    # Live spot at entry, not the signal bar's close (fixed
                    # 2026-07-22): the ±TARGET/STOP exit triggers measure spot
                    # movement from entry_spot, so it must be the spot at the
                    # moment we actually entered, not a bar close from up to 5
                    # minutes earlier that was never traded against.
                    try:
                        entry_spot = get_nifty_spot()
                    except Exception:
                        entry_spot = spot  # fall back to signal bar close, logged below
                        log.warning("Live spot fetch failed at bear entry — using signal bar close as entry_spot")
                    long_ltp, short_ltp = open_spread_legs(long_sym, short_sym, quantity)
                    net_debit = long_ltp - short_ltp
                    log.info(f"ENTRY BEAR PUT SPREAD {bear_signal} | +{long_sym} -{short_sym} | "
                             f"entry_spot={entry_spot:.2f} (signal bar close {spot:.2f}) | "
                             f"net_debit={net_debit:.2f} (sized against est {net_debit_est:.2f}, "
                             f"1-lot margin Rs {margin_per_lot:,.0f}) | qty={quantity} "
                             f"(~Rs {margin_per_lot * quantity / LOT_SIZE:,.0f} margin)")
                    if net_debit > net_debit_est * 1.5:
                        log.warning(f"Real fill net_debit {net_debit:.2f} is well above the {net_debit_est:.2f}pt "
                                    f"estimate used for sizing -- price moved between the sizing quote and the "
                                    f"fill; actual capital committed this trade (~Rs {net_debit * quantity:,.0f}) "
                                    f"may exceed the intended risk budget.")
                    state["bear_traded"]   = True
                    state["bear_position"] = {
                        "long_symbol": long_sym, "short_symbol": short_sym,
                        "entry_spot": entry_spot, "entry_net_debit": net_debit,
                        "entry_time": now.isoformat(), "signal": bear_signal,
                        "quantity": quantity,
                    }
            except Exception as e:
                log.error(f"Bear entry failed: {e}")

        # ── Enter Bull Call Spread ─────────────────────────────────────────────
        if bull_signal and big_trend_day:
            log.info(f"BULL signal {bull_signal} skipped — big-trend prior day "
                      f"({state['prev_move_pct']:.2f}% > {PREV_MOVE_THRESHOLD}%), only BEAR allowed "
                      f"through on big-trend days (direction-aware filter, 2026-08-04).")
        elif bull_signal and not state["bull_traded"] and not state["bull_position"]:
            try:
                net_debit_est, margin_per_lot, long_sym, short_sym = estimate_spread_capital_requirement("CE")
                quantity = compute_lot_quantity(net_debit_est, margin_per_lot)
                if quantity <= 0:
                    log.info(f"BULL signal {bull_signal} skipped — even 1 lot doesn't fit "
                             f"the per-leg capital pool (1-lot margin Rs {margin_per_lot:,.0f}, "
                             f"est net_debit {net_debit_est:.2f}). "
                             f"Not marking bull_traded -- will retry next tick.")
                else:
                    # Live spot at entry, not the signal bar's close — see the
                    # matching note at the bear entry above (fixed 2026-07-22).
                    try:
                        entry_spot = get_nifty_spot()
                    except Exception:
                        entry_spot = spot
                        log.warning("Live spot fetch failed at bull entry — using signal bar close as entry_spot")
                    long_ltp, short_ltp = open_spread_legs(long_sym, short_sym, quantity)
                    net_debit = long_ltp - short_ltp
                    log.info(f"ENTRY BULL CALL SPREAD {bull_signal} | +{long_sym} -{short_sym} | "
                             f"entry_spot={entry_spot:.2f} (signal bar close {spot:.2f}) | "
                             f"net_debit={net_debit:.2f} (sized against est {net_debit_est:.2f}, "
                             f"1-lot margin Rs {margin_per_lot:,.0f}) | qty={quantity} "
                             f"(~Rs {margin_per_lot * quantity / LOT_SIZE:,.0f} margin)")
                    if net_debit > net_debit_est * 1.5:
                        log.warning(f"Real fill net_debit {net_debit:.2f} is well above the {net_debit_est:.2f}pt "
                                    f"estimate used for sizing -- price moved between the sizing quote and the "
                                    f"fill; actual capital committed this trade (~Rs {net_debit * quantity:,.0f}) "
                                    f"may exceed the intended risk budget.")
                    state["bull_traded"]   = True
                    state["bull_position"] = {
                        "long_symbol": long_sym, "short_symbol": short_sym,
                        "entry_spot": entry_spot, "entry_net_debit": net_debit,
                        "entry_time": now.isoformat(), "signal": bull_signal,
                        "quantity": quantity,
                    }
            except Exception as e:
                log.error(f"Bull entry failed: {e}")

    except Exception as e:
        log.error(f"Signal detection failed: {e}")

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
        time.sleep(300)  # 5 minutes
