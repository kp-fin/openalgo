"""
nifty_credit_iron_condor_backtest.py -- intraday-only credit spread (iron condor)
backtest for weekly Nifty options. Reuses bs_pricing.py (BS premium, IV=15% flat,
same convention as the ORB_Spread family of rebuilt backtests).

Hypothesis under test: fixed-schedule, delta-anchored short strikes (no directional
signal), sell both a bull put spread and a bear call spread every trading day,
hold intraday only (flat by 15:15), stop at a multiple of credit received measured
on the spread's net value.

Data: Nifty daily OHLC (spot), 2021-07-01 -> 2026-07-29, pulled via OpenAlgo
/api/v1/history and cached to nifty_daily.json (see scratchpad). No real option-
chain data used -- premiums are BS-modeled off spot, so this is a MODELED backtest,
same caveat as bs_pricing.py's other consumers.

Approximation used for intraday entry/exit (daily bars only, no intraday data
pulled for this pass): entry premium priced off the day's OPEN (proxy for 09:45
entry), exit premium priced off the day's CLOSE (proxy for 15:15 exit). Stop-loss
is checked using the day's HIGH and LOW as the worst-case adverse spot excursion
for each side of the condor (put side stressed by the LOW, call side by the HIGH).
This is a simplification -- it assumes the stop-checking spot path reaches the
day's extreme before mean-reverting to the close, which is conservative (more
stop-outs, not fewer) rather than optimistic.
"""

import json
import sys
from datetime import date, datetime

sys.path.insert(0, r"D:\MyKnowledgeBase\FRIDAY - Super Agent\OpenAlgo\strategies\backtests")
from bs_pricing import bs_price, nearest_atm_strike, nearest_weekly_expiry  # noqa: E402

DATA_PATH = r"C:\Users\padhi\AppData\Local\Temp\claude\D--MyKnowledgeBase-FRIDAY---Super-Agent\bf835870-f057-4af6-97f7-014a096ae942\scratchpad\nifty_daily.json"

STRIKE_STEP = 50
TARGET_DELTA = 0.15          # short leg target delta (approx, via OTM-distance search)
HEDGE_WIDTH_PTS = 100         # long leg distance beyond short leg
STOP_MULTIPLE = 1.5           # exit if spread net value >= STOP_MULTIPLE x credit received
CAPITAL = 500_000.0
DEPLOYMENT_CAP_PCT = 0.80
LOT_SIZE = 65
RISK_FREE_RATE = 0.06
IV = 0.15


def load_bars():
    with open(DATA_PATH, encoding="utf-8-sig") as f:
        rows = json.load(f)
    bars = []
    for r in rows:
        d = datetime.fromisoformat(r["timestamp"]).date()
        bars.append({
            "date": d,
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
        })
    bars.sort(key=lambda b: b["date"])
    return bars


def find_short_strike_by_delta(spot, trade_date, expiry_date, option_type, target_delta):
    """Search OTM strikes stepping by STRIKE_STEP, approximate delta via a
    finite-difference bump on the BS price (no separate delta formula needed --
    consistent with using bs_price as the single pricing primitive)."""
    bump = 1.0
    best_strike, best_diff = None, None
    strike = nearest_atm_strike(spot, STRIKE_STEP)
    # search up to 30 strikes OTM
    for i in range(1, 31):
        k = strike + i * STRIKE_STEP if option_type == "CE" else strike - i * STRIKE_STEP
        if k <= 0:
            continue
        p_up = bs_price(spot + bump, k, trade_date, expiry_date, option_type, iv=IV, r=RISK_FREE_RATE)
        p_dn = bs_price(spot - bump, k, trade_date, expiry_date, option_type, iv=IV, r=RISK_FREE_RATE)
        delta = (p_up - p_dn) / (2 * bump)
        d = abs(delta) if option_type == "CE" else abs(delta)
        diff = abs(d - target_delta)
        if best_diff is None or diff < best_diff:
            best_diff, best_strike = diff, k
        if d < target_delta * 0.5:
            # moved well past target, stop searching further OTM
            break
    return best_strike


def spread_value(spot, short_k, long_k, trade_date, expiry_date, option_type):
    """Net value of the spread FROM THE SELLER'S PERSPECTIVE POV as a liability:
    short_premium - long_premium (this is what the seller would pay to close)."""
    short_p = bs_price(spot, short_k, trade_date, expiry_date, option_type, iv=IV, r=RISK_FREE_RATE)
    long_p = bs_price(spot, long_k, trade_date, expiry_date, option_type, iv=IV, r=RISK_FREE_RATE)
    return short_p - long_p


def run_backtest():
    bars = load_bars()
    trades = []

    for bar in bars:
        d = bar["date"]
        if d.weekday() >= 5:
            continue
        expiry = nearest_weekly_expiry(d)
        if expiry == d:
            continue  # skip expiry day itself (theta/gamma blowup, matches BB Squeeze convention)

        spot_open = bar["open"]
        spot_high = bar["high"]
        spot_low = bar["low"]
        spot_close = bar["close"]

        # --- Put side (bull put spread: sell higher strike PE, buy lower strike PE) ---
        short_pe = find_short_strike_by_delta(spot_open, d, expiry, "PE", TARGET_DELTA)
        long_pe = short_pe - HEDGE_WIDTH_PTS

        # --- Call side (bear call spread: sell lower strike CE, buy higher strike CE) ---
        short_ce = find_short_strike_by_delta(spot_open, d, expiry, "CE", TARGET_DELTA)
        long_ce = short_ce + HEDGE_WIDTH_PTS

        for side, short_k, long_k, opt in (("PUT", short_pe, long_pe, "PE"), ("CALL", short_ce, long_ce, "CE")):
            credit = spread_value(spot_open, short_k, long_k, d, expiry, opt)
            if credit <= 0:
                continue  # degenerate pricing, skip

            stop_value = credit * STOP_MULTIPLE
            max_loss_width = (short_k - long_k) if side == "PUT" else (long_k - short_k)
            max_loss_width = abs(max_loss_width) - credit

            # worst-case adverse excursion within the day for this side
            adverse_spot = spot_low if side == "PUT" else spot_high
            worst_value = spread_value(adverse_spot, short_k, long_k, d, expiry, opt)

            stopped = worst_value >= stop_value
            if stopped:
                exit_value = stop_value
                exit_reason = "STOP"
            else:
                exit_value = spread_value(spot_close, short_k, long_k, d, expiry, opt)
                exit_reason = "EOD"

            pnl_per_share = credit - exit_value  # seller: received credit, pays exit_value to close
            pnl_per_share = max(pnl_per_share, -max_loss_width)  # cap at width-defined max loss

            trades.append({
                "date": d.isoformat(),
                "side": side,
                "short_strike": short_k,
                "long_strike": long_k,
                "credit_pts": round(credit, 2),
                "exit_value_pts": round(exit_value, 2),
                "pnl_pts": round(pnl_per_share, 2),
                "pnl_rupees": round(pnl_per_share * LOT_SIZE, 2),
                "exit_reason": exit_reason,
                "spot_open": spot_open,
                "spot_close": spot_close,
            })

    return trades


def summarize(trades):
    import statistics

    n = len(trades)
    wins = [t for t in trades if t["pnl_rupees"] > 0]
    losses = [t for t in trades if t["pnl_rupees"] <= 0]
    win_rate = len(wins) / n * 100 if n else 0.0
    gross_profit = sum(t["pnl_rupees"] for t in wins)
    gross_loss = -sum(t["pnl_rupees"] for t in losses)
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    daily_pnl = {}
    for t in trades:
        daily_pnl.setdefault(t["date"], 0.0)
        daily_pnl[t["date"]] += t["pnl_rupees"]
    daily_series = list(daily_pnl.values())
    mean_daily = statistics.mean(daily_series) if daily_series else 0.0
    std_daily = statistics.pstdev(daily_series) if len(daily_series) > 1 else 0.0
    sharpe_annualized = (mean_daily / std_daily) * (252 ** 0.5) if std_daily > 0 else 0.0

    total_pnl = sum(t["pnl_rupees"] for t in trades)
    stopped_n = len([t for t in trades if t["exit_reason"] == "STOP"])

    print(f"Trades (legs): {n}")
    print(f"Trading days: {len(daily_pnl)}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Profit factor: {pf:.2f}")
    print(f"Total P&L: Rs {total_pnl:,.2f}")
    print(f"Stopped legs: {stopped_n} ({stopped_n/n*100:.1f}%)")
    print(f"Mean daily P&L: Rs {mean_daily:,.2f}  Std daily P&L: Rs {std_daily:,.2f}")
    print(f"Annualized Sharpe (daily P&L based, rf=0): {sharpe_annualized:.2f}")

    return {
        "n_legs": n,
        "n_days": len(daily_pnl),
        "win_rate": win_rate,
        "profit_factor": pf,
        "total_pnl": total_pnl,
        "sharpe": sharpe_annualized,
        "stopped_pct": stopped_n / n * 100 if n else 0,
    }


if __name__ == "__main__":
    trades = run_backtest()
    summary = summarize(trades)

    out_dir = r"D:\MyKnowledgeBase\FRIDAY - Super Agent\OpenAlgo\strategies\backtests\nifty_credit_iron_condor"
    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "trades.json"), "w", encoding="utf-8") as f:
        json.dump(trades, f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
