"""
bs_pricing.py — shared Black-Scholes option premium helper for the vault's rebuilt
backtests (BB Squeeze PE, PDH Breakout CE, VWAP Reclaim CE).

Matches the assumption stated in each strategy's spec: IV = 15% flat, simulation only
(no real option-chain data used — these strategies price a hypothetical ATM option off
the Nifty spot signal). Nearest weekly expiry, ATM = nearest 50-pt strike.

NOTE: the 3 strategies this module served (BB Squeeze PE, PDH Breakout CE, VWAP Reclaim CE)
were retired and deleted 2026-07-17 -- this module is now orphaned (no live consumer), kept
only as a reference in case a future strategy needs the same BS-pricing/expiry-lookup pattern.
The expiry helper below was corrected 2026-07-17 (NIFTY's weekly expiry moved Thursday -> Tuesday
since this module was originally written).
"""

from datetime import date, timedelta

import numpy as np
from scipy.stats import norm

RISK_FREE_RATE = 0.06   # standard assumption for INR short-dated options
IV_ASSUMED = 0.15        # per each strategy's spec


def nearest_atm_strike(spot: float, step: int = 50) -> float:
    return round(spot / step) * step


def nearest_weekly_expiry(trade_date: date) -> date:
    """Next Tuesday on/after trade_date (weekday() : Mon=0 ... Tue=1). Corrected
    2026-07-17 -- NIFTY's weekly expiry moved from Thursday to Tuesday; this
    function originally targeted Thursday (weekday()==3)."""
    days_ahead = (1 - trade_date.weekday()) % 7
    expiry = trade_date + timedelta(days=days_ahead)
    if expiry == trade_date:
        # today is Tuesday and IS the expiry — still valid as "nearest", callers
        # that need to skip same-day expiry (BB Squeeze) handle that themselves.
        return expiry
    return expiry


def bs_price(spot: float, strike: float, trade_date: date, expiry_date: date,
             option_type: str, iv: float = IV_ASSUMED, r: float = RISK_FREE_RATE) -> float:
    """
    European option premium via Black-Scholes. option_type: 'CE' or 'PE'.
    T is time-to-expiry in years using calendar days (min 1 day to avoid T=0 blowup
    on expiry day itself).
    """
    days_to_expiry = max((expiry_date - trade_date).days, 1)
    T = days_to_expiry / 365.0
    S, K, sigma = spot, strike, iv

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "CE":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "PE":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'CE' or 'PE', got {option_type!r}")

    return max(price, 0.05)  # floor — real premiums never price to exactly zero
